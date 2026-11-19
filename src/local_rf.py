"""Operaciones espaciales y Random Forest 100% en CPU/SEPAL sobre GeoTIFF locales.

Incluye:
- Muestreo rápido de las 24 bandas del Stack local en puntos de control (`rasterio.sample`).
- Propuesta de puntos candidatos de manglar desde `DinMangle` local erosionado (`scipy.ndimage`).
- Predicción espacial bloque a bloque de modelos `scikit-learn` (`n_jobs=-1`) a GeoTIFF `uint8`
  con cálculo exacto de superficie en hectáreas y mapa de 6 estratos de Acuerdo/Desacuerdo.
- Generador de diseño muestral estratificado tipo AREA2 / Olofsson (2014) directo en SEPAL
  (CSV `EPSG:4326` para Collect Earth Online) y cruce rápido de CEO contra los 4 GeoTIFFs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy.ndimage import binary_erosion
from shapely.geometry import Point


def extract_bands_at_points(
    stack_path: Path,
    gdf_points: gpd.GeoDataFrame,
    qa_path: Optional[Path] = None,
    require_base_aoi: bool = False,
) -> gpd.GeoDataFrame:
    """Extrae las 24 bandas del Stack local en cada punto (milisegundos, sin GEE)."""
    if gdf_points.empty:
        raise ValueError("El GeoDataFrame de puntos está vacío.")

    with rasterio.open(stack_path) as dstk:
        band_names = list(dstk.descriptions)
        if not band_names or len(band_names) != dstk.count:
            band_names = [f"B_{i}" for i in range(1, dstk.count + 1)]

        pts_utm = gdf_points.to_crs(dstk.crs)
        coords = [(geom.x, geom.y) for geom in pts_utm.geometry]
        sampled = np.vstack(list(dstk.sample(coords)))  # (N, 24)

    out = gdf_points.copy().reset_index(drop=True)
    for j, bname in enumerate(band_names):
        out[bname] = sampled[:, j].astype(np.float32)

    valid = np.all(np.isfinite(sampled), axis=1)

    if qa_path is not None and Path(qa_path).is_file():
        with rasterio.open(qa_path) as dqa:
            pts_qa = gdf_points.to_crs(dqa.crs)
            qa_vals = np.vstack(list(dqa.sample([(g.x, g.y) for g in pts_qa.geometry])))
            valid &= qa_vals[:, 4] == 1  # MASCARA_COMUN
            if require_base_aoi:
                valid &= qa_vals[:, 5] == 1  # MASCARA_AOI_BASE

    return out.loc[valid].reset_index(drop=True)


def propose_mangle_candidates_local(
    guia_tif_path: Path,
    qa_tif_path: Path,
    region: str,
    n_points: int = 120,
    mangle_val: int = 1,
    erosion_m: float = 20.0,
    seed: int = 20250822,
) -> gpd.GeoDataFrame:
    """Propone puntos candidatos dentro del núcleo erosionado de DinMangle usando NumPy/SciPy."""
    with rasterio.open(guia_tif_path) as dguia, rasterio.open(qa_tif_path) as dqa:
        guia_arr = dguia.read(1)
        m_comun = dqa.read(5) == 1
        m_base = dqa.read(6) == 1
        scale_m = abs(float(dguia.transform.a))
        erosion_px = max(0, int(round(erosion_m / max(scale_m, 1.0))))

        mangle_mask = (guia_arr == mangle_val) & m_comun & m_base
        if erosion_px > 0 and np.any(mangle_mask):
            struct = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=bool)
            mangle_mask = binary_erosion(mangle_mask, structure=struct) & m_comun & m_base

        rows, cols = np.where(mangle_mask)
        if rows.size == 0:
            raise RuntimeError(
                f"No se encontraron píxeles válidos de clase {mangle_val} en {guia_tif_path} tras erosión."
            )

        rng = np.random.default_rng(seed)
        k = min(int(n_points), int(rows.size))
        sel = rng.choice(rows.size, size=k, replace=False)
        sr, sc = rows[sel], cols[sel]

        xs, ys = rasterio.transform.xy(dguia.transform, sr, sc, offset="center")
        crs_src = dguia.crs

    transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    recs = [
        {
            "clase_id": 1,
            "clase": "Mangle",
            "fuente": "guia_DinMangle_local",
            "region": region.upper(),
            "geometry": Point(float(lon), float(lat)),
        }
        for lon, lat in zip(lons, lats)
    ]
    return gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")


def _build_uint8_colormap(class_colors: Dict[int, str]) -> Dict[int, tuple[int, int, int, int]]:
    cmap = {0: (0, 0, 0, 0)}
    for cid, hex_col in class_colors.items():
        h = hex_col.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        cmap[int(cid)] = (r, g, b, 255)
    return cmap


def predict_rf_to_geotiff(
    model: Any,
    stack_path: Path,
    qa_path: Path,
    feature_names: Sequence[str],
    out_tif_path: Path,
    class_colors: Dict[int, str],
    block_size: int = 1024,
    mask_base_only: bool = True,
) -> Dict[int, float]:
    """Clasifica el raster completo bloque a bloque en SEPAL y devuelve áreas en hectáreas."""
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = out_tif_path.with_suffix(".tmp.tif")

    # Asegurar máximo paralelismo en predicción scikit-learn
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1

    with rasterio.open(stack_path) as dstk, rasterio.open(qa_path) as dqa:
        desc = list(dstk.descriptions)
        feat_indices = [desc.index(f) + 1 for f in feature_names]
        height, width = dstk.height, dstk.width
        pixel_ha = abs(float(dstk.transform.a) * float(dstk.transform.e)) / 10000.0

        counts: Dict[int, int] = {}
        with rasterio.open(
            tmp_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=dstk.crs,
            transform=dstk.transform,
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst_out:
            for r0 in range(0, height, block_size):
                h = min(block_size, height - r0)
                for c0 in range(0, width, block_size):
                    w = min(block_size, width - c0)
                    win = Window(c0, r0, w, h)

                    m_valid = dqa.read(5, window=win) == 1
                    if mask_base_only:
                        m_valid &= dqa.read(6, window=win) == 1

                    out_blk = np.zeros((h, w), dtype=np.uint8)
                    if np.any(m_valid):
                        blk = dstk.read(feat_indices, window=win)  # (F, h, w)
                        m_finite = m_valid & np.all(np.isfinite(blk), axis=0)
                        if np.any(m_finite):
                            x_mat = np.moveaxis(blk[:, m_finite], 0, 1)  # (N_valid, F)
                            preds = model.predict(x_mat).astype(np.uint8)
                            out_blk[m_finite] = preds
                            u_vals, u_cnt = np.unique(preds, return_counts=True)
                            for uv, uc in zip(u_vals, u_cnt):
                                counts[int(uv)] = counts.get(int(uv), 0) + int(uc)

                    dst_out.write(out_blk, 1, window=win)

            dst_out.set_band_description(1, "clasificacion")
            dst_out.write_colormap(1, _build_uint8_colormap(class_colors))

    tmp_tif.replace(out_tif_path)
    return {cid: float(cnt * pixel_ha) for cid, cnt in sorted(counts.items())}


def build_agreement_stratum_geotiff(
    map_ms_tif: Path,
    map_sar_tif: Path,
    map_comb_tif: Path,
    out_tif_path: Path,
    block_size: int = 1024,
) -> Dict[int, float]:
    """Genera mapa de 6 estratos de Acuerdo/Desacuerdo en Mangle para muestreo AREA2/CEO."""
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = out_tif_path.with_suffix(".tmp.tif")

    with rasterio.open(map_ms_tif) as dms, rasterio.open(map_sar_tif) as dsar, rasterio.open(map_comb_tif) as dcomb:
        height, width = dcomb.height, dcomb.width
        pixel_ha = abs(float(dcomb.transform.a) * float(dcomb.transform.e)) / 10000.0
        counts: Dict[int, int] = {}

        with rasterio.open(
            tmp_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=dcomb.crs,
            transform=dcomb.transform,
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            for r0 in range(0, height, block_size):
                h = min(block_size, height - r0)
                win = Window(0, r0, width, h)
                a_ms = dms.read(1, window=win)
                a_sar = dsar.read(1, window=win)
                a_comb = dcomb.read(1, window=win)

                valid = (a_ms > 0) & (a_sar > 0) & (a_comb > 0)
                sum_m = (a_ms == 1).astype(np.uint8) + (a_sar == 1).astype(np.uint8) + (a_comb == 1).astype(np.uint8)
                desacuerdo = valid & (sum_m > 0) & (sum_m < 3)
                est = np.where(valid, a_comb, 0).astype(np.uint8)
                est[desacuerdo] = 6
                dst.write(est, 1, window=win)

                u_vals, u_cnt = np.unique(est[valid], return_counts=True)
                for uv, uc in zip(u_vals, u_cnt):
                    counts[int(uv)] = counts.get(int(uv), 0) + int(uc)

            dst.set_band_description(1, "estrato_acuerdo_desacuerdo")
            dst.write_colormap(
                1,
                _build_uint8_colormap(
                    {
                        1: "#006837",
                        2: "#78c679",
                        3: "#2b83ba",
                        4: "#d7191c",
                        5: "#fdae61",
                        6: "#984ea3",
                    }
                ),
            )

    tmp_tif.replace(out_tif_path)
    return {sid: float(cnt * pixel_ha) for sid, cnt in sorted(counts.items())}


def generate_stratified_sample_ceo(
    stratum_tif_path: Path,
    out_csv_path: Path,
    out_gpkg_path: Optional[Path] = None,
    stratum_areas_ha: Optional[Dict[int, float]] = None,
    se_target_oa: float = 0.015,
    min_per_stratum: int = 50,
    total_n: Optional[int] = None,
    seed: int = 20250822,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Genera diseño muestral estratificado tipo Olofsson (2014) / AREA2 directo en SEPAL.

    Calcula el tamaño muestral según Olofsson et al. (2014, Eq. 13), asigna cuotas
    combinando proporción de área + mínimo por estrato (para proteger clases raras),
    y extrae puntos aleatorios sin reemplazo en `EPSG:4326` listos para Collect Earth Online.
    """
    with rasterio.open(stratum_tif_path) as ds:
        arr = ds.read(1)
        crs_src = ds.crs
        transform = ds.transform
        pixel_ha = abs(float(transform.a) * float(transform.e)) / 10000.0

    strata_ids = sorted(int(v) for v in np.unique(arr) if v > 0)
    if not strata_ids:
        raise ValueError(f"El mapa de estratos '{stratum_tif_path}' no tiene píxeles válidos > 0.")

    if stratum_areas_ha is None:
        stratum_areas_ha = {s: float(np.sum(arr == s) * pixel_ha) for s in strata_ids}
    else:
        stratum_areas_ha = {int(k): float(v) for k, v in stratum_areas_ha.items() if int(k) in strata_ids}

    a_tot = sum(stratum_areas_ha.values())
    w_map = {s: stratum_areas_ha[s] / a_tot for s in strata_ids}

    # Precisión de usuario esperada conservadora por estrato (0.75 en estrato de desacuerdo, 0.88 en acuerdo)
    expected_u = {s: (0.75 if s == 6 else 0.88) for s in strata_ids}
    s_i = {s: math.sqrt(expected_u[s] * (1.0 - expected_u[s])) for s in strata_ids}

    if total_n is None:
        num = sum(w_map[s] * s_i[s] for s in strata_ids)
        n_calc = int(math.ceil((num / max(se_target_oa, 1e-4)) ** 2))
        total_n = max(n_calc, len(strata_ids) * min_per_stratum)

    # Asignar mínimo garantizado por estrato y repartir el remanente proporcionalmente a W_i * S_i
    n_avail = {s: int(np.sum(arr == s)) for s in strata_ids}
    alloc = {s: min(min_per_stratum, n_avail[s]) for s in strata_ids}
    rem = max(0, total_n - sum(alloc.values()))
    if rem > 0:
        weights_neym = np.array([w_map[s] * s_i[s] for s in strata_ids], dtype=float)
        weights_neym = weights_neym / max(weights_neym.sum(), 1e-12)
        extra = np.floor(weights_neym * rem).astype(int)
        for s, ex_n in zip(strata_ids, extra):
            alloc[s] = min(n_avail[s], alloc[s] + int(ex_n))
        # Completar residuo por redondeo
        leftover = total_n - sum(alloc.values())
        for s in sorted(strata_ids, key=lambda x: w_map[x], reverse=True):
            if leftover <= 0:
                break
            if alloc[s] < n_avail[s]:
                add_n = min(leftover, n_avail[s] - alloc[s])
                alloc[s] += add_n
                leftover -= add_n

    rng = np.random.default_rng(seed)
    transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)

    rows_pts = []
    rows_plan = []
    for s in strata_ids:
        ys, xs = np.where(arr == s)
        n_s = alloc[s]
        sel = rng.choice(ys.size, size=n_s, replace=False)
        px_x, px_y = rasterio.transform.xy(transform, ys[sel], xs[sel], offset="center")
        lons, lats = transformer.transform(px_x, px_y)
        for lon, lat in zip(lons, lats):
            rows_pts.append({"stratum": int(s), "lon": float(lon), "lat": float(lat), "geometry": Point(float(lon), float(lat))})
        rows_plan.append(
            {
                "estrato_id": s,
                "area_ha": stratum_areas_ha[s],
                "W_i": w_map[s],
                "n_muestras": n_s,
            }
        )

    gdf_pts = gpd.GeoDataFrame(rows_pts, geometry="geometry", crs="EPSG:4326")
    # Mezclar orden espacial para interpretación ciega en Collect Earth Online
    gdf_pts = gdf_pts.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    gdf_pts.insert(0, "PLOTID", np.arange(1, len(gdf_pts) + 1))
    gdf_pts.insert(1, "LON", gdf_pts["lon"])
    gdf_pts.insert(2, "LAT", gdf_pts["lat"])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_pts[["PLOTID", "LON", "LAT", "stratum"]].to_csv(out_csv_path, index=False)
    if out_gpkg_path is not None:
        out_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_pts[["PLOTID", "stratum", "lon", "lat", "geometry"]].to_file(out_gpkg_path, driver="GPKG")

    plan_df = pd.DataFrame(rows_plan)
    return gdf_pts, plan_df


def cross_points_with_local_maps(
    df_pts: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    map_paths: Dict[str, Path],
) -> pd.DataFrame:
    """Cruza coordenadas EPSG:4326 (CEO) contra los GeoTIFFs clasificados locales en SEPAL (< 1 s)."""
    df_out = df_pts.copy().reset_index(drop=True)
    gdf_4326 = gpd.GeoDataFrame(
        df_out[[lon_col, lat_col]],
        geometry=[Point(float(x), float(y)) for x, y in zip(df_out[lon_col], df_out[lat_col])],
        crs="EPSG:4326",
    )

    for model_name, tif_path in map_paths.items():
        with rasterio.open(tif_path) as ds:
            pts_proj = gdf_4326.to_crs(ds.crs)
            coords = [(g.x, g.y) for g in pts_proj.geometry]
            vals = np.array([int(v[0]) for v in ds.sample(coords)], dtype=int)
        df_out[f"pred_{model_name}"] = np.where(vals > 0, vals, np.nan)

    pred_cols = [f"pred_{m}" for m in map_paths]
    df_out = df_out.dropna(subset=pred_cols).reset_index(drop=True)
    for col in pred_cols:
        df_out[col] = df_out[col].astype(int)
    return df_out
