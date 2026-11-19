"""Preprocesamiento Sentinel-1 ARD 100% local en SEPAL (sin GEE).

Soporta dos fuentes configurables en `config.yaml`:
1. `OPERA_RTC_S1` (predeterminado): productos oficiales NASA/JPL `OPERA_L2_RTC-S1_V1`
   vía `asf_search`, ya en `gamma0` potencia lineal con corrección radiométrica de terreno
   (Copernicus GLO-30 DEM) en COGs ligeros (~8 MB/burst). Sobre ellos aplicamos
   **Refined Lee 7×7 direccional en potencia lineal** vectorizado en NumPy/SciPy,
   conversión segura a decibelios (piso `1e-6` = -60 dB) y mediana estacional en SEPAL.
2. `GRD_SNAP` (opcional, 10 m nativos): descarga escenas Sentinel-1 IW `GRD_HD` vía
   `asf_search` y ejecuta la cadena oficial ESA SNAP (`gpt`) localmente en la VM de SEPAL
   (`Apply-Orbit-File -> ThermalNoiseRemoval -> Remove-GRD-Border-Noise -> Calibration
   -> Terrain-Flattening -> Refined Lee -> Terrain-Correction 10 m UTM -> dB`).
"""

from __future__ import annotations

import netrc
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import asf_search as asf
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import convolve

from .config_utils import resolve_max_workers
from .local_aoi import rasterize_aoi_masks


# ---------------------------------------------------------------------------
# 1. Autenticacion NASA Earthdata Login & deteccion SNAP GPT
# ---------------------------------------------------------------------------
def save_earthdata_netrc(username: str, password: str, path: Optional[Path] = None) -> Path:
    """Guarda credenciales de NASA Earthdata Login en ~/.netrc con permisos 0600."""
    target = Path(path) if path else Path.home() / ".netrc"
    host = "urs.earthdata.nasa.gov"
    entry = f"machine {host}\nlogin {username}\npassword {password}\n"

    existing = ""
    if target.is_file():
        lines = target.read_text(encoding="utf-8").splitlines()
        filtered = []
        skip_block = False
        for ln in lines:
            if ln.strip().startswith("machine "):
                skip_block = host in ln
            if not skip_block:
                filtered.append(ln)
        existing = "\n".join(filtered).strip()

    content = (existing + "\n\n" + entry).strip() + "\n"
    target.write_text(content, encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def has_earthdata_credentials() -> bool:
    """Verifica si hay credenciales o token configurados para NASA Earthdata / ASF."""
    if os.environ.get("EARTHDATA_TOKEN"):
        return True
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return True
    netrc_file = Path.home() / ".netrc"
    if netrc_file.is_file():
        try:
            auth = netrc.netrc(str(netrc_file)).authenticators("urs.earthdata.nasa.gov")
            return auth is not None
        except Exception:
            return False
    return False


def get_earthdata_session(
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
) -> asf.ASFSession:
    """Crea una sesión autenticada de asf_search sin exponer contraseñas."""
    tok = token or os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return asf.ASFSession().auth_with_token(tok)

    usr = username or os.environ.get("EARTHDATA_USERNAME")
    pwd = password or os.environ.get("EARTHDATA_PASSWORD")
    if usr and pwd:
        return asf.ASFSession().auth_with_creds(usr, pwd)

    netrc_file = Path.home() / ".netrc"
    if netrc_file.is_file():
        auth = netrc.netrc(str(netrc_file)).authenticators("urs.earthdata.nasa.gov")
        if auth is not None:
            return asf.ASFSession().auth_with_creds(auth[0], auth[2])

    raise RuntimeError(
        "No se encontraron credenciales de NASA Earthdata Login (~/.netrc ni variables EARTHDATA_*). "
        "Ejecuta la celda de credenciales de Earthdata en el Notebook 00 o 01."
    )


def detect_snap_gpt() -> Optional[str]:
    """Detecta el ejecutable `gpt` de ESA SNAP en la instancia de SEPAL."""
    candidates = [
        shutil.which("gpt"),
        "/usr/local/snap/bin/gpt",
        "/opt/snap/bin/gpt",
        str(Path.home() / "snap" / "bin" / "gpt"),
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# 2. Inventario y busqueda de orbitas en ASF (no requiere autenticacion)
# ---------------------------------------------------------------------------
def _exclusive_end_iso(fin_exclusiva: str) -> str:
    t1 = pd.Timestamp(fin_exclusiva) - pd.Timedelta(seconds=1)
    return t1.strftime("%Y-%m-%dT%H:%M:%SZ")


def search_asf_s1(
    wkt_4326: str,
    inicio: str,
    fin_exclusiva: str,
    fuente: str = "OPERA_RTC_S1",
    orbit_pass: Optional[str] = None,
):
    """Busca productos Sentinel-1 (OPERA-S1 RTC o SENTINEL-1 GRD_HD) en NASA ASF."""
    fuente_u = fuente.upper()
    kwargs: Dict[str, Any] = {
        "intersectsWith": wkt_4326,
        "start": f"{inicio}T00:00:00Z",
        "end": _exclusive_end_iso(fin_exclusiva),
    }
    if orbit_pass:
        kwargs["flightDirection"] = orbit_pass.upper()

    if fuente_u == "OPERA_RTC_S1":
        kwargs["dataset"] = asf.DATASET.OPERA_S1
        kwargs["processingLevel"] = asf.PRODUCT_TYPE.RTC
    elif fuente_u == "GRD_SNAP":
        kwargs["dataset"] = asf.DATASET.SENTINEL1
        kwargs["processingLevel"] = asf.PRODUCT_TYPE.GRD_HD
        kwargs["beamMode"] = asf.BEAMMODE.IW
    else:
        raise ValueError(f"Fuente S1 desconocida: '{fuente}'. Usa 'OPERA_RTC_S1' o 'GRD_SNAP'.")

    return asf.search(**kwargs)


def inspect_s1_orbits_asf(
    wkt_4326: str,
    inicio: str,
    fin_exclusiva: str,
    fuente: str = "OPERA_RTC_S1",
) -> Dict[str, Dict[str, Any]]:
    """Diagnostica escenas/bursts disponibles en ASCENDING vs DESCENDING sin requerir clave."""
    res = search_asf_s1(wkt_4326, inicio, fin_exclusiva, fuente=fuente, orbit_pass=None)
    summary: Dict[str, Dict[str, Any]] = {}
    for orb in ("ASCENDING", "DESCENDING"):
        sub = [p for p in res if str(p.properties.get("flightDirection", "")).upper() == orb]
        fechas = sorted({str(p.properties.get("startTime", ""))[:10] for p in sub if p.properties.get("startTime")})
        tracks = sorted({int(p.properties["pathNumber"]) for p in sub if p.properties.get("pathNumber") is not None})
        summary[orb] = {
            "n_productos": len(sub),
            "n_fechas_unicas": len(fechas),
            "orbitas_relativas": tracks,
        }
    return summary


# ---------------------------------------------------------------------------
# 3. Filtro Refined Lee 7x7 vectorizado en potencia lineal (NumPy / SciPy)
# ---------------------------------------------------------------------------
def _window_mean_and_var(img: np.ndarray, ok: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula media y varianza local insesgada sobre píxeles válidos usando convolución exacta."""
    x = np.where(ok, img, 0.0)
    k = kernel.astype(np.float64)
    cnt = convolve(ok.astype(np.float64), k, mode="constant", cval=0.0)
    s1 = convolve(x, k, mode="constant", cval=0.0)
    s2 = convolve(x * x, k, mode="constant", cval=0.0)

    with np.errstate(all="ignore"):
        mean = np.where(cnt > 0.5, s1 / np.maximum(cnt, 1.0), np.nan)
        ex2 = np.where(cnt > 0.5, s2 / np.maximum(cnt, 1.0), np.nan)
        raw_var = np.maximum(ex2 - mean * mean, 0.0)
        var = np.where(cnt > 1.5, raw_var * (cnt / np.maximum(cnt - 1.0, 1.0)), 0.0)
    return mean, var


def refined_lee_7x7_numpy(img_lin: np.ndarray) -> np.ndarray:
    """Aplica filtro direccional Refined Lee 7x7 en potencia lineal preservando bordes.

    Implementación vectorizada en NumPy/SciPy fiel a Lee (1981) / Mullissa et al. (2021):
    - Evalúa 9 sub-ventanas 3x3 espaciadas cada 2 px dentro de la ventana 7x7.
    - Selecciona la orientación homogénea dominante entre 8 direcciones (rectángulos + diagonales).
    - Estima el coeficiente MMSE de ponderación `b in [0, 1]` a partir de las 5 sub-ventanas
      con menor varianza normalizada de speckle.
    """
    arr = np.asarray(img_lin, dtype=np.float64)
    ok = np.isfinite(arr) & (arr > 0.0)
    if not np.any(ok):
        return np.full_like(arr, np.nan, dtype=np.float32)

    h, w = arr.shape
    k3 = np.ones((3, 3), dtype=np.float64)
    mean3, var3 = _window_mean_and_var(arr, ok, k3)

    # Extraer 9 sub-ventanas 3x3 centradas en offsets (-2, 0, +2)
    m3_pad = np.pad(np.where(np.isfinite(mean3), mean3, 0.0), 2, mode="edge")
    v3_pad = np.pad(np.where(np.isfinite(var3), var3, 0.0), 2, mode="edge")

    offsets = [
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2),  (0, 0),  (0, 2),
        (2, -2),  (2, 0),  (2, 2),
    ]
    s_mean = np.stack([m3_pad[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w] for dy, dx in offsets], axis=0)
    s_var = np.stack([v3_pad[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w] for dy, dx in offsets], axis=0)

    # 4 gradientes direccionales
    grads = np.stack(
        [
            np.abs(s_mean[1] - s_mean[7]),
            np.abs(s_mean[6] - s_mean[2]),
            np.abs(s_mean[3] - s_mean[5]),
            np.abs(s_mean[0] - s_mean[8]),
        ],
        axis=0,
    )
    max_g = np.max(grads, axis=0)

    # 8 direcciones posibles (1..8) según borde dominante
    d_candidates = np.stack(
        [
            np.where((s_mean[1] - s_mean[4]) > (s_mean[4] - s_mean[7]), 1, 5),
            np.where((s_mean[6] - s_mean[4]) > (s_mean[4] - s_mean[2]), 2, 6),
            np.where((s_mean[3] - s_mean[4]) > (s_mean[4] - s_mean[5]), 3, 7),
            np.where((s_mean[0] - s_mean[4]) > (s_mean[4] - s_mean[8]), 4, 8),
        ],
        axis=0,
    )
    best_axis = np.argmax(grads == max_g[None, :, :], axis=0)
    directions = np.take_along_axis(d_candidates, best_axis[None, :, :], axis=0)[0]

    # Ruido speckle local: media de las 5 sub-ventanas más homogéneas
    sample_stats = s_var / np.maximum(s_mean * s_mean, 1e-12)
    sigma_v = np.mean(np.sort(sample_stats, axis=0)[:5], axis=0)

    # 8 kernels direccionales 7x7 (rectángulo + diagonal y sus 3 rotaciones de 90 grados)
    rect0 = np.vstack([np.zeros((3, 7), dtype=np.float64), np.ones((4, 7), dtype=np.float64)])
    diag0 = np.tril(np.ones((7, 7), dtype=np.float64))

    dir_kernels = {
        1: rect0,
        2: diag0,
        3: np.rot90(rect0, 1),
        4: np.rot90(diag0, 1),
        5: np.rot90(rect0, 2),
        6: np.rot90(diag0, 2),
        7: np.rot90(rect0, 3),
        8: np.rot90(diag0, 3),
    }

    dir_mean = np.zeros((h, w), dtype=np.float64)
    dir_var = np.zeros((h, w), dtype=np.float64)
    for d_id, kern in dir_kernels.items():
        mask_d = (directions == d_id) & ok
        if not np.any(mask_d):
            continue
        dm, dv = _window_mean_and_var(arr, ok, kern)
        dir_mean[mask_d] = np.nan_to_num(dm[mask_d], nan=0.0)
        dir_var[mask_d] = np.nan_to_num(dv[mask_d], nan=0.0)

    var_x = (dir_var - (dir_mean * dir_mean) * sigma_v) / (sigma_v + 1.0)
    b_weight = np.clip(var_x / np.maximum(dir_var, 1e-12), 0.0, 1.0)
    filtered = dir_mean + b_weight * (arr - dir_mean)

    return np.where(ok, filtered, np.nan).astype(np.float32)


def apply_local_speckle_filter(img_lin: np.ndarray, method: str = "REFINED_LEE", kernel_size: int = 7) -> np.ndarray:
    """Aplica filtro de moteado configurado ('REFINED_LEE', 'BOXCAR' o 'NONE') en potencia lineal."""
    m = method.upper().replace(" ", "_")
    if m == "NONE":
        return np.where(np.isfinite(img_lin) & (img_lin > 0), img_lin, np.nan).astype(np.float32)
    if m == "REFINED_LEE":
        return refined_lee_7x7_numpy(img_lin)
    if m == "BOXCAR":
        ok = np.isfinite(img_lin) & (img_lin > 0)
        mean_b, _ = _window_mean_and_var(np.asarray(img_lin, dtype=np.float64), ok, np.ones((kernel_size, kernel_size)))
        return np.where(ok, mean_b, np.nan).astype(np.float32)
    raise ValueError(f"Filtro speckle no soportado: '{method}'.")


# ---------------------------------------------------------------------------
# 4. Pipeline OPERA RTC-S1 en Python puro (predeterminado en SEPAL)
# ---------------------------------------------------------------------------
def _extract_opera_urls(prod: Any) -> Dict[str, str]:
    """Obtiene URLs de _VV.tif, _VH.tif y _mask.tif desde un producto OPERA-S1."""
    urls = [prod.properties.get("url", "")] + list(prod.properties.get("additionalUrls", []) or [])
    out: Dict[str, str] = {}
    for u in urls:
        if not u:
            continue
        if u.endswith("_VV.tif"):
            out["VV"] = u
        elif u.endswith("_VH.tif"):
            out["VH"] = u
        elif u.endswith("_mask.tif"):
            out["MASK"] = u
    return out


def _read_warped_to_grid(path: Path, target_crs: str, target_transform: Affine, width: int, height: int, resampling: Resampling, dtype: str) -> np.ndarray:
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=target_crs,
            transform=target_transform,
            width=width,
            height=height,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1).astype(dtype)


def _process_opera_date_group(
    date_str: str,
    prods: List[Any],
    url_to_local: Dict[str, Path],
    grid_info: Dict[str, Any],
    cfg_s1: Dict[str, Any],
    out_dir: Path,
) -> Optional[Path]:
    """Mosaica bursts OPERA de la misma fecha, aplica Refined Lee (con margen de kernel) y convierte a dB."""
    pass_tif = out_dir / f"s1_pass_{date_str}.tif"
    height = grid_info["height"]
    width = grid_info["width"]
    crs = grid_info["crs"]
    transform = grid_info["transform"]

    if pass_tif.is_file():
        with rasterio.open(pass_tif) as ds:
            if ds.count == 2 and ds.width == width and ds.height == height and str(ds.crs) == str(crs):
                return pass_tif

    # Expandir grilla por el margen de kernel (ej. 60 m = 6 px) para cero efecto de borde
    pad_px = max(4, int(round(float(grid_info.get("margen_kernel_m", 60.0)) / float(grid_info["scale_m"]))))
    ext_transform = transform * Affine.translation(-pad_px, -pad_px)
    ext_w = width + 2 * pad_px
    ext_h = height + 2 * pad_px

    speckle = cfg_s1.get("speckle_filter", "REFINED_LEE")
    k_size = int(cfg_s1.get("kernel_size", 7))
    piso = float(cfg_s1.get("piso_lineal_min", 1e-6))

    vv_frames, vh_frames = [], []
    for p in prods:
        u_map = _extract_opera_urls(p)
        if "VV" not in u_map or "VH" not in u_map:
            continue
        vv_p = url_to_local.get(u_map["VV"])
        vh_p = url_to_local.get(u_map["VH"])
        if not vv_p or not vh_p or not vv_p.is_file() or not vh_p.is_file():
            continue

        vv_arr = _read_warped_to_grid(vv_p, crs, ext_transform, ext_w, ext_h, Resampling.bilinear, "float32")
        vh_arr = _read_warped_to_grid(vh_p, crs, ext_transform, ext_w, ext_h, Resampling.bilinear, "float32")
        valid = np.isfinite(vv_arr) & (vv_arr > 0) & np.isfinite(vh_arr) & (vh_arr > 0)

        if "MASK" in u_map and u_map["MASK"] in url_to_local and url_to_local[u_map["MASK"]].is_file():
            m_arr = _read_warped_to_grid(url_to_local[u_map["MASK"]], crs, ext_transform, ext_w, ext_h, Resampling.nearest, "uint8")
            # En OPERA RTC-S1: 0 = píxel válido sin layover ni sombra; 1/2/3 = layover/shadow; 255 = fill
            valid &= (m_arr == 0)

        vv_frames.append(np.where(valid, vv_arr, np.nan))
        vh_frames.append(np.where(valid, vh_arr, np.nan))

    if not vv_frames:
        return None

    with np.errstate(all="ignore"):
        vv_lin = vv_frames[0] if len(vv_frames) == 1 else np.nanmean(np.stack(vv_frames, axis=0), axis=0)
        vh_lin = vh_frames[0] if len(vh_frames) == 1 else np.nanmean(np.stack(vh_frames, axis=0), axis=0)

    if not np.any(np.isfinite(vv_lin) & np.isfinite(vh_lin)):
        return None

    # Aplicar Refined Lee en potencia lineal y recortar a la grilla final sin margen
    vv_f = apply_local_speckle_filter(vv_lin, method=speckle, kernel_size=k_size)[pad_px : pad_px + height, pad_px : pad_px + width]
    vh_f = apply_local_speckle_filter(vh_lin, method=speckle, kernel_size=k_size)[pad_px : pad_px + height, pad_px : pad_px + width]

    valid_out = np.isfinite(vv_f) & np.isfinite(vh_f)
    vv_db = np.where(valid_out, 10.0 * np.log10(np.maximum(vv_f, piso)), np.nan).astype(np.float32)
    vh_db = np.where(valid_out, 10.0 * np.log10(np.maximum(vh_f, piso)), np.nan).astype(np.float32)

    tmp = pass_tif.with_suffix(".tmp.tif")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(vv_db, 1)
        dst.write(vh_db, 2)
    tmp.replace(pass_tif)
    return pass_tif


# ---------------------------------------------------------------------------
# 5. Soporte opcional GRD + ESA SNAP `gpt` (si esta instalado en SEPAL)
# ---------------------------------------------------------------------------
def build_snap_grd_ard_xml(xml_path: Path, epsg_crs: str, scale_m: float = 10.0, speckle_filter: str = "Refined Lee") -> Path:
    """Genera un grafo XML estándar de ESA SNAP para Sentinel-1 IW GRD -> Gamma0 dB UTM."""
    wkt_crs = rasterio.crs.CRS.from_string(epsg_crs).to_wkt()
    xml_text = f"""<graph id="S1_GRD_ARD">
  <version>1.0</version>
  <node id="Read"><operator>Read</operator><sources/><parameters><file>${{input}}</file></parameters></node>
  <node id="Apply-Orbit-File"><operator>Apply-Orbit-File</operator><sources><sourceProduct refid="Read"/></sources><parameters><orbitType>Sentinel Precise (Auto Download)</orbitType><continueOnFail>true</continueOnFail></parameters></node>
  <node id="ThermalNoiseRemoval"><operator>ThermalNoiseRemoval</operator><sources><sourceProduct refid="Apply-Orbit-File"/></sources><parameters><removeThermalNoise>true</removeThermalNoise></parameters></node>
  <node id="Remove-GRD-Border-Noise"><operator>Remove-GRD-Border-Noise</operator><sources><sourceProduct refid="ThermalNoiseRemoval"/></sources><parameters><borderLimit>500</borderLimit><trimThreshold>0.5</trimThreshold></parameters></node>
  <node id="Calibration"><operator>Calibration</operator><sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources><parameters><outputBetaBand>true</outputBetaBand><selectedPolarisations>VH,VV</selectedPolarisations></parameters></node>
  <node id="Terrain-Flattening"><operator>Terrain-Flattening</operator><sources><sourceProduct refid="Calibration"/></sources><parameters><demName>Copernicus 30m Global DEM</demName><demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod></parameters></node>
  <node id="Speckle-Filter"><operator>Speckle-Filter</operator><sources><sourceProduct refid="Terrain-Flattening"/></sources><parameters><filter>{speckle_filter}</filter></parameters></node>
  <node id="Terrain-Correction"><operator>Terrain-Correction</operator><sources><sourceProduct refid="Speckle-Filter"/></sources><parameters><demName>Copernicus 30m Global DEM</demName><demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod><imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod><pixelSpacingInMeter>{scale_m}</pixelSpacingInMeter><mapProjection>{wkt_crs}</mapProjection><nodataValueAtSea>false</nodataValueAtSea></parameters></node>
  <node id="LinearToFromdB"><operator>LinearToFromdB</operator><sources><sourceProduct refid="Terrain-Correction"/></sources><parameters/></node>
  <node id="Write"><operator>Write</operator><sources><sourceProduct refid="LinearToFromdB"/></sources><parameters><file>${{output}}</file><formatName>GeoTIFF-BigTIFF</formatName></parameters></node>
</graph>
"""
    xml_path.write_text(xml_text, encoding="utf-8")
    return xml_path


def _process_grd_snap_scene(zip_path: Path, snap_gpt: str, xml_path: Path, grid_info: Dict[str, Any], out_tif: Path, n_threads: int) -> Optional[Path]:
    if out_tif.is_file():
        return out_tif
    raw_snap_tif = out_tif.with_suffix(".snap_raw.tif")
    if not raw_snap_tif.is_file():
        cmd = [snap_gpt, str(xml_path), f"-Pinput={zip_path}", f"-Poutput={raw_snap_tif}", "-q", str(max(2, n_threads))]
        subprocess.run(cmd, check=True)

    # Reproyectar al grid exacto (banda 1=VH, 2=VV según SNAP -> reordenamos a VV, VH)
    with rasterio.open(raw_snap_tif) as src:
        desc = [d or "" for d in src.descriptions]
        idx_vv = next((i + 1 for i, d in enumerate(desc) if "VV" in d.upper()), 2 if src.count >= 2 else 1)
        idx_vh = next((i + 1 for i, d in enumerate(desc) if "VH" in d.upper()), 1)
        with WarpedVRT(
            src,
            crs=grid_info["crs"],
            transform=grid_info["transform"],
            width=grid_info["width"],
            height=grid_info["height"],
            resampling=Resampling.bilinear,
        ) as vrt:
            vv = vrt.read(idx_vv).astype(np.float32)
            vh = vrt.read(idx_vh).astype(np.float32)

    vv[~np.isfinite(vv) | (vv == 0)] = np.nan
    vh[~np.isfinite(vh) | (vh == 0)] = np.nan
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=grid_info["height"],
        width=grid_info["width"],
        count=2,
        dtype="float32",
        crs=grid_info["crs"],
        transform=grid_info["transform"],
        nodata=np.nan,
        compress="DEFLATE",
        tiled=True,
    ) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)
    raw_snap_tif.unlink(missing_ok=True)
    return out_tif


# ---------------------------------------------------------------------------
# 6. Orquestador principal de composicion estacional S1 en SEPAL
# ---------------------------------------------------------------------------
def build_local_s1_composite(
    region: str,
    temporada: str,
    grid_info: Dict[str, Any],
    inicio: str,
    fin_exclusiva: str,
    orbit_pass: str,
    cfg: Dict[str, Any],
    session: Optional[asf.ASFSession] = None,
) -> Dict[str, Any]:
    """Descarga y procesa Sentinel-1 ARD localmente en SEPAL y genera mosaico mediano en dB."""
    cfg_s1 = cfg["s1"]
    fuente = str(cfg_s1.get("fuente_local", "OPERA_RTC_S1")).upper()
    reg_l = region.lower()
    temp_u = temporada.upper()

    if session is None:
        session = get_earthdata_session()

    cache_root = Path(cfg.get("computo", {}).get("cache_dir", "/tmp/tesis_manglar_cache"))
    raw_cache = cache_root / "s1_raw" / reg_l / temp_u.lower()
    pass_cache = cache_root / "s1_passes" / reg_l / temp_u.lower()
    comp_dir = cache_root / "composites"
    for d in (raw_cache, pass_cache, comp_dir):
        d.mkdir(parents=True, exist_ok=True)

    out_s1 = comp_dir / f"s1_{reg_l}_{temp_u.lower()}.tif"
    out_nobs = comp_dir / f"nobs_s1_{reg_l}_{temp_u.lower()}.tif"

    results = search_asf_s1(
        grid_info["wkt_search_4326"],
        inicio,
        fin_exclusiva,
        fuente=fuente,
        orbit_pass=orbit_pass,
    )
    if not results:
        raise RuntimeError(
            f"No se encontraron productos S1 ({fuente}) en ASF para {region} {temp_u} ({orbit_pass})."
        )

    workers = resolve_max_workers(cfg, cap=12)
    pass_paths: List[Path] = []
    inv_rows: List[Dict[str, Any]] = []

    if fuente == "OPERA_RTC_S1":
        # Filtrar productos que tengan tanto _VV.tif como _VH.tif
        urls_needed: List[str] = []
        url_to_local: Dict[str, Path] = {}
        by_date: Dict[str, List[Any]] = defaultdict(list)

        for prod in results:
            u_map = _extract_opera_urls(prod)
            if "VV" not in u_map or "VH" not in u_map:
                continue
            dt = str(prod.properties.get("startTime", ""))[:10]
            by_date[dt].append(prod)
            for key in ("VV", "VH", "MASK"):
                if key in u_map:
                    u = u_map[key]
                    fname = u.split("/")[-1]
                    lp = raw_cache / fname
                    url_to_local[u] = lp
                    if not lp.is_file():
                        urls_needed.append(u)
            inv_rows.append(
                {
                    "region": region.upper(),
                    "temporada": temp_u,
                    "fecha": dt,
                    "granule": prod.properties.get("fileID", prod.properties.get("sceneName")),
                    "plataforma": prod.properties.get("platform"),
                    "orbita": prod.properties.get("flightDirection"),
                    "orbita_relativa": prod.properties.get("pathNumber"),
                    "burst_id": prod.properties.get("operaBurstID"),
                }
            )

        urls_needed = list(dict.fromkeys(urls_needed))
        if urls_needed:
            asf.download_urls(urls=urls_needed, path=str(raw_cache), session=session, processes=workers)

        for dt in sorted(by_date.keys()):
            p_tif = _process_opera_date_group(dt, by_date[dt], url_to_local, grid_info, cfg_s1, pass_cache)
            if p_tif is not None:
                pass_paths.append(p_tif)

    elif fuente == "GRD_SNAP":
        snap_gpt = detect_snap_gpt()
        if not snap_gpt:
            raise RuntimeError(
                "Configuraste fuente_local='GRD_SNAP', pero no se encontró ESA SNAP `gpt` en esta instancia. "
                "Usa fuente_local='OPERA_RTC_S1' en config.yaml (recomendado en Python puro)."
            )
        xml_path = build_snap_grd_ard_xml(pass_cache / f"grd_ard_{reg_l}.xml", grid_info["crs"], float(grid_info["scale_m"]))
        # Descargar GRD_HD ZIPs en paralelo
        results.download(path=str(raw_cache), session=session, processes=min(4, workers))
        for prod in results:
            fname = prod.properties.get("fileName")
            if not fname:
                continue
            zip_p = raw_cache / fname
            out_t = pass_cache / f"{Path(fname).stem}_ard.tif"
            p_tif = _process_grd_snap_scene(zip_p, snap_gpt, xml_path, grid_info, out_t, workers)
            if p_tif is not None:
                pass_paths.append(p_tif)
            inv_rows.append(
                {
                    "region": region.upper(),
                    "temporada": temp_u,
                    "fecha": str(prod.properties.get("startTime", ""))[:10],
                    "granule": prod.properties.get("sceneName"),
                    "plataforma": prod.properties.get("platform"),
                    "orbita": prod.properties.get("flightDirection"),
                    "orbita_relativa": prod.properties.get("pathNumber"),
                }
            )

    if not pass_paths:
        raise RuntimeError(f"No se generó ninguna pasada S1 válida para {region} {temp_u}.")

    # Ensamblaje mediano temporal por bloques y recorte exacto al AOI_BUFFER
    height, width = grid_info["height"], grid_info["width"]
    chunk_rows = int(cfg.get("computo", {}).get("block_size", 1024))
    _, mask_buf = rasterize_aoi_masks(grid_info)
    band_names = [f"VV_{temp_u}", f"VH_{temp_u}"]

    readers = [rasterio.open(p) for p in pass_paths]
    try:
        with rasterio.open(
            out_s1,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=2,
            dtype="float32",
            crs=grid_info["crs"],
            transform=grid_info["transform"],
            nodata=np.nan,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst_img, rasterio.open(
            out_nobs,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint16",
            crs=grid_info["crs"],
            transform=grid_info["transform"],
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst_nobs:
            for r0 in range(0, height, chunk_rows):
                h = min(chunk_rows, height - r0)
                win = Window(0, r0, width, h)
                cube = np.stack([ds.read(window=win) for ds in readers], axis=0)  # (T, 2, h, w)
                with np.errstate(all="ignore"):
                    med = np.nanmedian(cube, axis=0).astype(np.float32)
                mb = mask_buf[r0 : r0 + h, :]
                med[:, ~mb] = np.nan
                nobs = np.where(mb, np.sum(np.isfinite(cube[:, 0, :, :]), axis=0), 0).astype(np.uint16)
                dst_img.write(med, window=win)
                dst_nobs.write(nobs, 1, window=win)

            dst_img.set_band_description(1, band_names[0])
            dst_img.set_band_description(2, band_names[1])
            dst_nobs.set_band_description(1, f"NOBS_S1_{temp_u}")
    finally:
        for r in readers:
            r.close()

    inv_df = pd.DataFrame(inv_rows)
    tracks = sorted({int(x) for x in inv_df["orbita_relativa"].dropna().unique()}) if not inv_df.empty else []

    return {
        "image_path": out_s1,
        "nobs_path": out_nobs,
        "band_names": band_names,
        "fuente": fuente,
        "n_productos": len(inv_rows),
        "n_fechas_procesadas": len(pass_paths),
        "orbitas_relativas": tracks,
        "inventario": inv_df,
    }
