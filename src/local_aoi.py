"""Gestión local de AOIs, grillas UTM exactas a 10 m y descarga puntual única de DinMangle."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import box

from .config_utils import ensure_output_dirs, init_gee, resolve_max_workers


def get_local_aoi_path(region: str, cfg: Dict[str, Any]) -> Path:
    dirs = ensure_output_dirs(cfg)
    return dirs["aoi"] / f"AOI_{region.upper()}.gpkg"


def get_local_guia_path(region: str, cfg: Dict[str, Any]) -> Path:
    dirs = ensure_output_dirs(cfg)
    return dirs["guia"] / f"DinMangle_2020_2022_V23_{region.lower()}.tif"


def ensure_local_aoi(region: str, cfg: Dict[str, Any], overwrite: bool = False) -> gpd.GeoDataFrame:
    """Carga el AOI local en EPSG:4326; si no existe, lo descarga una sola vez desde GEE Asset."""
    region_key = region.upper()
    dirs = ensure_output_dirs(cfg)
    gpkg_path = get_local_aoi_path(region_key, cfg)

    # También permitir que el usuario coloque manualmente un .shp o .geojson con ese nombre
    candidates = [
        gpkg_path,
        dirs["aoi"] / f"AOI_{region_key}.geojson",
        dirs["aoi"] / f"AOI_{region_key}.shp",
    ]
    if not overwrite:
        for cand in candidates:
            if cand.is_file():
                gdf = gpd.read_file(cand)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                return gdf.to_crs("EPSG:4326")

    # Descarga ligera única (< 1 s) desde el Asset de GEE configurado en config.yaml
    import ee

    init_gee(cfg)
    asset_id = cfg["assets"]["aoi"][region_key]
    info = ee.data.getAsset(asset_id)
    if info.get("type") == "IMAGE":
        fc = ee.FeatureCollection([ee.Feature(ee.Image(asset_id).geometry(), {"nombre": region_key})])
    else:
        fc = ee.FeatureCollection(asset_id)

    geojson = fc.getInfo()
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if gdf.empty:
        raise ValueError(f"El asset AOI '{asset_id}' no contiene geometrías.")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"[OK] AOI descargado una sola vez y guardado localmente en: {gpkg_path}")
    return gdf


def compute_region_grid(
    aoi_gdf_4326: gpd.GeoDataFrame,
    crs: str,
    scale_m: float = 10.0,
    buffer_m: float = 100.0,
    margen_kernel_m: float = 60.0,
) -> Dict[str, Any]:
    """Calcula la grilla UTM exacta (10 m) ajustada al bbox de AOI + buffer_m."""
    aoi_4326 = aoi_gdf_4326.to_crs("EPSG:4326")
    aoi_utm = aoi_4326.to_crs(crs)
    union_base = (
        aoi_utm.geometry.union_all()
        if hasattr(aoi_utm.geometry, "union_all")
        else aoi_utm.geometry.unary_union
    )
    union_buffer = union_base.buffer(float(buffer_m))
    union_proc = union_buffer.buffer(float(margen_kernel_m))

    minx, miny, maxx, maxy = union_buffer.bounds
    xmin = math.floor(minx / scale_m) * scale_m
    ymin = math.floor(miny / scale_m) * scale_m
    xmax = math.ceil(maxx / scale_m) * scale_m
    ymax = math.ceil(maxy / scale_m) * scale_m

    width = int(round((xmax - xmin) / scale_m))
    height = int(round((ymax - ymin) / scale_m))
    transform = Affine(float(scale_m), 0.0, float(xmin), 0.0, -float(scale_m), float(ymax))

    gdf_buf_4326 = gpd.GeoDataFrame(geometry=[union_buffer], crs=crs).to_crs("EPSG:4326")
    gdf_proc_4326 = gpd.GeoDataFrame(geometry=[union_proc], crs=crs).to_crs("EPSG:4326")
    b4326 = gdf_proc_4326.total_bounds  # (minlon, minlat, maxlon, maxlat)

    # Polígono WKT simplificado (< 200 vértices, válido para búsquedas en NASA ASF Search)
    hull_simple = gdf_proc_4326.geometry.iloc[0].simplify(0.002, preserve_topology=True)
    if hull_simple.geom_type == "MultiPolygon":
        hull_simple = hull_simple.convex_hull
    wkt_search = hull_simple.wkt

    return {
        "crs": crs,
        "scale_m": float(scale_m),
        "buffer_m": float(buffer_m),
        "margen_kernel_m": float(margen_kernel_m),
        "transform": transform,
        "width": width,
        "height": height,
        "bounds_utm": (xmin, ymin, xmax, ymax),
        "bounds_4326": tuple(float(v) for v in b4326),
        "bounds_stack_4326": tuple(float(v) for v in gdf_buf_4326.total_bounds),
        "wkt_search_4326": wkt_search,
        "geom_base_utm": union_base,
        "geom_buffer_utm": union_buffer,
        "geom_proc_utm": union_proc,
        "aoi_base_4326": aoi_4326,
    }


def rasterize_aoi_masks(grid_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Rasteriza máscaras booleanas `MASCARA_AOI_BASE` y `MASCARA_AOI_BUFFER` en la grilla UTM."""
    shape = (grid_info["height"], grid_info["width"])
    transform = grid_info["transform"]
    mask_base = geometry_mask(
        [grid_info["geom_base_utm"]],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    mask_buffer = geometry_mask(
        [grid_info["geom_buffer_utm"]],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return mask_base, mask_buffer


def _grid_matches(path: Path, grid_info: Dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    with rasterio.open(path) as ds:
        return (
            str(ds.crs) == str(grid_info["crs"])
            and ds.width == grid_info["width"]
            and ds.height == grid_info["height"]
            and tuple(ds.transform)[:6] == tuple(grid_info["transform"])[:6]
        )


def ensure_local_mangle_guide(
    region: str,
    cfg: Dict[str, Any],
    grid_info: Dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Obtiene el recorte local uint8 (1 banda) de `DinMangle_2020_2022_V23` alineado a la grilla.

    Si ya existe localmente, no se conecta a GEE. Si no existe, descarga en bloques pequeños
    mediante `ee.data.computePixels` (~3–10 s, sin colas ni Google Drive).
    """
    out_path = get_local_guia_path(region, cfg)
    if not overwrite and _grid_matches(out_path, grid_info):
        return out_path

    import ee

    init_gee(cfg)
    guia_id = cfg["assets"]["mangle_guia"]
    info = ee.data.getAsset(guia_id)
    if info.get("type") == "TABLE":
        img = ee.Image(0).byte().paint(ee.FeatureCollection(guia_id), 1).rename("b1").unmask(0).toUint8()
    else:
        img = ee.Image(guia_id).select(0).rename("b1").unmask(0).toUint8()

    width = grid_info["width"]
    height = grid_info["height"]
    transform = grid_info["transform"]
    crs = grid_info["crs"]
    tile_sz = 2048  # 2048 x 2048 x 1 byte = 4 MB (< límite 48 MB de computePixels)

    tiles = []
    for row0 in range(0, height, tile_sz):
        for col0 in range(0, width, tile_sz):
            h = min(tile_sz, height - row0)
            w = min(tile_sz, width - col0)
            win_t = transform * Affine.translation(col0, row0)
            tiles.append((row0, col0, h, w, win_t))

    out_arr = np.zeros((height, width), dtype=np.uint8)

    def _fetch_tile(spec):
        row0, col0, h, w, win_t = spec
        req = {
            "expression": img,
            "fileFormat": "NUMPY_NDARRAY",
            "bandIds": ["b1"],
            "grid": {
                "dimensions": {"width": w, "height": h},
                "affineTransform": {
                    "scaleX": float(win_t.a),
                    "shearX": 0.0,
                    "translateX": float(win_t.c),
                    "shearY": 0.0,
                    "scaleY": float(win_t.e),
                    "translateY": float(win_t.f),
                },
                "crsCode": crs,
            },
        }
        arr = ee.data.computePixels(req)
        return row0, col0, h, w, arr["b1"].astype(np.uint8)

    workers = min(8, resolve_max_workers(cfg))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_tile, t) for t in tiles]
        for fut in as_completed(futs):
            r0, c0, h, w, data = fut.result()
            out_arr[r0 : r0 + h, c0 : c0 + w] = data

    # Aplicar recorte limpio a AOI_BUFFER
    _, mask_buf = rasterize_aoi_masks(grid_info)
    out_arr[~mask_buf] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(out_arr, 1)
        dst.set_band_description(1, "DinMangle_2020_2022_V23")

    print(f"[OK] Capa guía DinMangle guardada localmente en: {out_path}")
    return out_path
