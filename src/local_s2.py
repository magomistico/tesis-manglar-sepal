"""Preprocesamiento Sentinel-2 L2A 100% local en SEPAL vía AWS Earth Search STAC (sin GEE).

Lee únicamente las ventanas COG que intersectan el AOI desde la colección armonizada
`sentinel-2-c1-l2a` de Element84 Earth Search (acceso público gratuito), aplica detección
automática de offset BOA (+1000 / -0.1), enmascara nubes y sombras usando `SCL` + banda
`cloud` con protección de canales mareales/manglar inundado mediante `MNDWI`, y calcula
la mediana estacional por bloques para mantener consumo mínimo de RAM en la VM de SEPAL.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

from .config_utils import resolve_max_workers
from .local_aoi import rasterize_aoi_masks

# Mapeo oficial Protocolo -> Asset keys de AWS Earth Search (sentinel-2-c1-l2a)
S2_STAC_BAND_MAP: Dict[str, str] = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "rededge1",
    "B6": "rededge2",
    "B7": "rededge3",
    "B8": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}


def _exclusive_stac_datetime(inicio: str, fin_exclusiva: str) -> str:
    t0 = pd.Timestamp(inicio)
    t1 = pd.Timestamp(fin_exclusiva) - pd.Timedelta(seconds=1)
    return f"{t0.strftime('%Y-%m-%dT00:00:00Z')}/{t1.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def search_s2_stac(
    bounds_4326: Tuple[float, float, float, float],
    inicio: str,
    fin_exclusiva: str,
    cfg_s2: Dict[str, Any],
) -> List[Any]:
    """Busca escenas Sentinel-2 Collection 1 L2A en Earth Search STAC."""
    stac_url = cfg_s2.get("stac_url", "https://earth-search.aws.element84.com/v1")
    collection = cfg_s2.get("stac_collection", "sentinel-2-c1-l2a")
    max_cloud = float(cfg_s2.get("nubosidad_escena_max", 45))

    catalog = Client.open(stac_url)
    dt_range = _exclusive_stac_datetime(inicio, fin_exclusiva)
    search = catalog.search(
        collections=[collection],
        bbox=list(bounds_4326),
        datetime=dt_range,
        query={"eo:cloud_cover": {"lte": max_cloud}},
    )
    items = list(search.items())

    # Deduplicar por id manteniendo la versión más reciente
    by_id: Dict[str, Any] = {}
    for it in items:
        prev = by_id.get(it.id)
        if prev is None or str(it.properties.get("updated", "")) > str(prev.properties.get("updated", "")):
            by_id[it.id] = it
    return sorted(by_id.values(), key=lambda x: str(x.properties.get("datetime", "")))


def _resolve_scale_offset(asset_extra: Dict[str, Any], red_dn: Optional[np.ndarray] = None, scl: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Determina scale y offset de reflectancia con verificación física anti doble-offset."""
    rb = asset_extra.get("raster:bands") or [{}]
    scale = float(rb[0].get("scale", 0.0001))
    offset = float(rb[0].get("offset", 0.0))

    # Verificación física sobre agua o valores oscuros: evita restar 0.1 dos veces
    if offset != 0.0 and red_dn is not None:
        valid = red_dn > 0
        if scl is not None and np.any((scl == 6) & valid):
            sample = red_dn[(scl == 6) & valid].astype(np.float32) * scale
        elif np.any(valid):
            vals = red_dn[valid].astype(np.float32) * scale
            sample = vals[vals <= np.nanpercentile(vals, 5)]
        else:
            sample = np.array([], dtype=np.float32)

        if sample.size > 0:
            med_sin_offset = float(np.nanmedian(sample))
            if med_sin_offset + offset < -0.02:
                # El raster ya venía sin offset (+1000); ignorar offset del metadata
                offset = 0.0

    return scale, offset


def compute_s2_clear_mask(
    b3_refl: np.ndarray,
    b11_refl: np.ndarray,
    scl: np.ndarray,
    cloud_prob: Optional[np.ndarray],
    valid_data_mask: np.ndarray,
    cfg_s2: Dict[str, Any],
) -> np.ndarray:
    """Genera máscara booleana de píxeles limpios (True = válido) protegiendo manglar inundado."""
    scl_invalidos = set(int(c) for c in cfg_s2.get("scl_invalidos", [0, 1, 3, 8, 9, 10]))
    prob_max = float(cfg_s2.get("probabilidad_nube", 30))
    umbral_mndwi = float(cfg_s2.get("umbral_mndwi_agua", 0.10))
    buffer_px = int(cfg_s2.get("buffer_nubes_px", 2))

    # MNDWI + SCL 6 protege canales estuarinos y manglar con lámina de agua de ser tratados como sombra
    denom = b3_refl + b11_refl
    mndwi = np.where(np.abs(denom) > 1e-6, (b3_refl - b11_refl) / np.where(np.abs(denom) > 1e-6, denom, 1.0), 0.0)
    water_protected = (scl == 6) | (mndwi >= umbral_mndwi)

    scl_cloud = np.isin(scl, [8, 9, 10])
    prob_cloud = (cloud_prob > prob_max) if cloud_prob is not None else np.zeros_like(scl_cloud, dtype=bool)
    clouds_raw = scl_cloud | prob_cloud

    scl_bad = np.isin(scl, list(scl_invalidos)) | clouds_raw
    # Sobre agua protegida solo rechazamos nubes reales o píxeles saturados/nodata, NO falsas sombras (3)
    water_bad = np.isin(scl, [0, 1, 8, 9, 10]) | ((cloud_prob > max(prob_max, 50.0)) if cloud_prob is not None else False)
    mask_bad = np.where(water_protected, water_bad, scl_bad)

    if buffer_px > 0 and np.any(mask_bad):
        struct = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), dtype=bool)
        dilated = binary_dilation(mask_bad, structure=struct)
        # No propagar bordes dilatados de sombra sobre agua estuarina despejada
        mask_bad = np.where(water_protected, water_bad, dilated)

    return valid_data_mask & (~mask_bad)


def _gdal_env():
    return rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.jp2",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MAX_RETRY="4",
        GDAL_HTTP_RETRY_DELAY="2",
    )


def _read_cog_to_grid(href: str, grid_info: Dict[str, Any], resampling: Resampling, dtype: str) -> np.ndarray:
    with _gdal_env():
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=grid_info["crs"],
                transform=grid_info["transform"],
                width=grid_info["width"],
                height=grid_info["height"],
                resampling=resampling,
            ) as vrt:
                return vrt.read(1).astype(dtype)


def _process_and_cache_s2_scene(
    item: Any,
    band_names: List[str],
    grid_info: Dict[str, Any],
    mask_buffer: np.ndarray,
    cfg_s2: Dict[str, Any],
    cache_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Pre-filtra por SCL y descarga al caché local las 10 bandas recortadas y enmascaradas."""
    scene_tif = cache_dir / f"{item.id}.tif"
    grid_meta = (str(grid_info["crs"]), grid_info["width"], grid_info["height"], tuple(grid_info["transform"])[:6])
    if scene_tif.is_file():
        try:
            with rasterio.open(scene_tif) as ds:
                if (
                    ds.count == len(band_names)
                    and str(ds.crs) == grid_meta[0]
                    and ds.width == grid_meta[1]
                    and ds.height == grid_meta[2]
                ):
                    return {
                        "id": item.id,
                        "datetime": str(item.properties.get("datetime", ""))[:19],
                        "tile": item.properties.get("grid:code", item.properties.get("s2:mgrs_tile", "")),
                        "eo_cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
                        "path": scene_tif,
                        "cached": True,
                    }
        except Exception:
            scene_tif.unlink(missing_ok=True)

    assets = item.assets
    if "scl" not in assets:
        return None

    # Pre-chequeo económico leyendo SOLO SCL (20 m) sobre el AOI
    try:
        scl = _read_cog_to_grid(assets["scl"].href, grid_info, Resampling.nearest, "uint8")
    except Exception:
        return None

    scl_bad_set = set(int(c) for c in cfg_s2.get("scl_invalidos", [0, 1, 3, 8, 9, 10]))
    valid_in_aoi = mask_buffer & (~np.isin(scl, list(scl_bad_set)))
    valid_frac = float(valid_in_aoi.sum() / max(int(mask_buffer.sum()), 1))
    if valid_frac < 0.02:
        return None

    # Leer banda opcional `cloud` (probabilidad de nube C1)
    cloud_prob = None
    if "cloud" in assets:
        try:
            cloud_prob = _read_cog_to_grid(assets["cloud"].href, grid_info, Resampling.nearest, "float32")
        except Exception:
            cloud_prob = None

    # Leer las 10 bandas espectrales en el orden del protocolo
    raw_bands: Dict[str, np.ndarray] = {}
    scales_offsets: Dict[str, Dict[str, Any]] = {}
    for b_prot in band_names:
        stac_key = S2_STAC_BAND_MAP[b_prot]
        if stac_key not in assets:
            return None
        arr_dn = _read_cog_to_grid(assets[stac_key].href, grid_info, Resampling.bilinear, "float32")
        raw_bands[b_prot] = arr_dn
        scales_offsets[b_prot] = (
            assets[stac_key].extra_fields if hasattr(assets[stac_key], "extra_fields") else {}
        )

    # Resolver scale/offset usando banda roja B4 y SCL como control físico
    scale_ref, offset_ref = _resolve_scale_offset(scales_offsets["B4"], raw_bands.get("B4"), scl)

    refl_stack = np.full((len(band_names), grid_info["height"], grid_info["width"]), np.nan, dtype=np.float32)
    valid_dn = np.ones((grid_info["height"], grid_info["width"]), dtype=bool)
    for i, b_prot in enumerate(band_names):
        dn = raw_bands[b_prot]
        s_i, o_i = _resolve_scale_offset(scales_offsets[b_prot])
        # Alinear comportamiento del offset con la verificación física en B4
        if offset_ref == 0.0 and o_i != 0.0:
            o_i = 0.0
        elif offset_ref != 0.0 and o_i == 0.0:
            o_i = offset_ref
        valid_dn &= dn > 0
        refl = np.clip(dn * s_i + o_i, 0.0, 1.5)
        refl_stack[i] = refl

    idx_b3 = band_names.index("B3")
    idx_b11 = band_names.index("B11")
    clear = compute_s2_clear_mask(
        refl_stack[idx_b3],
        refl_stack[idx_b11],
        scl,
        cloud_prob,
        mask_buffer & valid_dn,
        cfg_s2,
    )

    if int(clear.sum()) == 0:
        return None

    refl_stack[:, ~clear] = np.nan

    tmp_tif = scene_tif.with_suffix(".tmp.tif")
    with rasterio.open(
        tmp_tif,
        "w",
        driver="GTiff",
        height=grid_info["height"],
        width=grid_info["width"],
        count=len(band_names),
        dtype="float32",
        crs=grid_info["crs"],
        transform=grid_info["transform"],
        nodata=np.nan,
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(refl_stack)
    tmp_tif.replace(scene_tif)

    return {
        "id": item.id,
        "datetime": str(item.properties.get("datetime", ""))[:19],
        "tile": item.properties.get("grid:code", item.properties.get("s2:mgrs_tile", "")),
        "eo_cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
        "valid_frac_aoi": round(float(clear.sum() / max(int(mask_buffer.sum()), 1)), 4),
        "path": scene_tif,
        "cached": False,
    }


def build_local_s2_composite(
    region: str,
    temporada: str,
    grid_info: Dict[str, Any],
    inicio: str,
    fin_exclusiva: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Construye el mosaico mediano estacional Sentinel-2 localmente en SEPAL."""
    cfg_s2 = cfg["s2"]
    band_names = list(cfg_s2["bandas"])
    reg_l = region.lower()
    temp_u = temporada.upper()

    cache_root = Path(cfg.get("computo", {}).get("cache_dir", "/tmp/tesis_manglar_cache"))
    scene_cache = cache_root / "s2" / reg_l / temp_u.lower()
    out_dir = cache_root / "composites"
    scene_cache.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_s2 = out_dir / f"s2_{reg_l}_{temp_u.lower()}.tif"
    out_nobs = out_dir / f"nobs_s2_{reg_l}_{temp_u.lower()}.tif"

    items = search_s2_stac(grid_info["bounds_4326"], inicio, fin_exclusiva, cfg_s2)
    if not items:
        raise RuntimeError(
            f"No se encontraron escenas Sentinel-2 en STAC para {region} {temp_u} ({inicio} a {fin_exclusiva})."
        )

    _, mask_buffer = rasterize_aoi_masks(grid_info)
    workers = resolve_max_workers(cfg, cap=12)

    valid_scenes: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _process_and_cache_s2_scene,
                it,
                band_names,
                grid_info,
                mask_buffer,
                cfg_s2,
                scene_cache,
            )
            for it in items
        ]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                valid_scenes.append(res)

    valid_scenes.sort(key=lambda r: r["datetime"])
    if not valid_scenes:
        raise RuntimeError(
            f"Ninguna escena Sentinel-2 tuvo píxeles despejados sobre {region} {temp_u}."
        )

    height = grid_info["height"]
    width = grid_info["width"]
    n_bands = len(band_names)
    chunk_rows = int(cfg.get("computo", {}).get("block_size", 1024))
    scene_paths = [s["path"] for s in valid_scenes]
    out_band_names = [f"{b}_{temp_u}" for b in band_names]

    # Abrir readers una vez y procesar por franjas horizontales para RAM controlada
    readers = [rasterio.open(p) for p in scene_paths]
    try:
        with rasterio.open(
            out_s2,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_bands,
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
                cube = np.stack([ds.read(window=win) for ds in readers], axis=0)  # (T, B, h, w)
                # Mediana ignorando NaN en el eje temporal
                with np.errstate(all="ignore"):
                    med = np.nanmedian(cube, axis=0).astype(np.float32)
                nobs = np.sum(np.isfinite(cube[:, 0, :, :]), axis=0).astype(np.uint16)
                dst_img.write(med, window=win)
                dst_nobs.write(nobs, 1, window=win)

            for i, name in enumerate(out_band_names, start=1):
                dst_img.set_band_description(i, name)
            dst_nobs.set_band_description(1, f"NOBS_S2_{temp_u}")
    finally:
        for r in readers:
            r.close()

    inv_df = pd.DataFrame(
        [
            {k: v for k, v in s.items() if k != "path"}
            for s in valid_scenes
        ]
    )

    return {
        "image_path": out_s2,
        "nobs_path": out_nobs,
        "band_names": out_band_names,
        "n_candidatas_stac": len(items),
        "n_escenas_validas": len(valid_scenes),
        "tiles": sorted({str(s.get("tile", "")) for s in valid_scenes if s.get("tile")}),
        "inventario": inv_df,
    }
