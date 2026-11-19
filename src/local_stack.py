"""Ensamblaje, control de calidad y visualización local del Stack Biestacional S1 + S2 (24b + 6 QA) en SEPAL."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from .config_utils import ensure_output_dirs, get_predictor_band_names
from .local_aoi import compute_region_grid, ensure_local_aoi, rasterize_aoi_masks
from .local_s1 import build_local_s1_composite
from .local_s2 import build_local_s2_composite


def get_local_stack_paths(region: str, cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Rutas estandarizadas en disco local de SEPAL para Stack, QA e inventarios."""
    dirs = ensure_output_dirs(cfg)
    reg_l = region.lower()
    ver = cfg["proyecto"]["version_pipeline"].lower()
    return {
        "stack": dirs["stack"] / f"stack_biestacional_{reg_l}_2025_{ver}.tif",
        "qa": dirs["stack"] / f"qa_stack_biestacional_{reg_l}_2025_{ver}.tif",
        "diagnostico": dirs["tablas"] / f"diagnostico_stack_{reg_l}.csv",
        "inventario_s2": dirs["tablas"] / f"inventario_s2_{reg_l}.csv",
        "inventario_s1": dirs["tablas"] / f"inventario_s1_{reg_l}.csv",
    }


def assemble_local_biseasonal_stack(
    s2_seca_path: Path,
    nobs_s2_seca_path: Path,
    s2_lluvia_path: Path,
    nobs_s2_lluvia_path: Path,
    s1_seca_path: Path,
    nobs_s1_seca_path: Path,
    s1_lluvia_path: Path,
    nobs_s1_lluvia_path: Path,
    mask_base: np.ndarray,
    mask_buffer: np.ndarray,
    band_names_24: Sequence[str],
    out_stack_path: Path,
    out_qa_path: Path,
    block_rows: int = 1024,
    metadata_tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """Une los 4 compuestos estacionales en el Stack Biestacional de 24 bandas + QA de 6 bandas."""
    out_stack_path.parent.mkdir(parents=True, exist_ok=True)
    out_qa_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        rasterio.open(s2_seca_path) as ds_s2s,
        rasterio.open(s2_lluvia_path) as ds_s2l,
        rasterio.open(s1_seca_path) as ds_s1s,
        rasterio.open(s1_lluvia_path) as ds_s1l,
        rasterio.open(nobs_s2_seca_path) as ds_n2s,
        rasterio.open(nobs_s2_lluvia_path) as ds_n2l,
        rasterio.open(nobs_s1_seca_path) as ds_n1s,
        rasterio.open(nobs_s1_lluvia_path) as ds_n1l,
    ):
        height, width = ds_s2s.height, ds_s2s.width
        crs = ds_s2s.crs
        transform = ds_s2s.transform

        for ds in (ds_s2l, ds_s1s, ds_s1l):
            if ds.width != width or ds.height != height or ds.crs != crs:
                raise ValueError("Las grillas de S1 y S2 no coinciden exactamente al ensamblar el stack.")

        n_bands = len(band_names_24)
        tmp_stack = out_stack_path.with_suffix(".tmp.tif")
        tmp_qa = out_qa_path.with_suffix(".tmp.tif")

        with rasterio.open(
            tmp_stack,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_bands,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst_stack, rasterio.open(
            tmp_qa,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=6,
            dtype="uint16",
            crs=crs,
            transform=transform,
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst_qa:
            for r0 in range(0, height, block_rows):
                h = min(block_rows, height - r0)
                win = Window(0, r0, width, h)

                a_s2s = ds_s2s.read(window=win)
                a_s2l = ds_s2l.read(window=win)
                a_s1s = ds_s1s.read(window=win)
                a_s1l = ds_s1l.read(window=win)

                stack_blk = np.concatenate([a_s2s, a_s2l, a_s1s, a_s1l], axis=0).astype(np.float32)
                mbuf = mask_buffer[r0 : r0 + h, :]
                mbase = mask_base[r0 : r0 + h, :]

                # Máscara común estricta: válido en todas las 24 bandas y dentro de AOI+buffer
                common = mbuf & np.all(np.isfinite(stack_blk), axis=0)
                stack_blk[:, ~common] = np.nan
                dst_stack.write(stack_blk, window=win)

                n2s = ds_n2s.read(1, window=win)
                n2l = ds_n2l.read(1, window=win)
                n1s = ds_n1s.read(1, window=win)
                n1l = ds_n1l.read(1, window=win)

                qa_blk = np.stack(
                    [
                        np.where(mbuf, n2s, 0).astype(np.uint16),
                        np.where(mbuf, n2l, 0).astype(np.uint16),
                        np.where(mbuf, n1s, 0).astype(np.uint16),
                        np.where(mbuf, n1l, 0).astype(np.uint16),
                        common.astype(np.uint16),
                        mbase.astype(np.uint16),
                    ],
                    axis=0,
                )
                dst_qa.write(qa_blk, window=win)

            for i, bname in enumerate(band_names_24, start=1):
                dst_stack.set_band_description(i, bname)
            if metadata_tags:
                dst_stack.update_tags(**metadata_tags)

            qa_names = [
                "NOBS_S2_SECA",
                "NOBS_S2_LLUVIA",
                "NOBS_S1_SECA",
                "NOBS_S1_LLUVIA",
                "MASCARA_COMUN",
                "MASCARA_AOI_BASE",
            ]
            for i, qname in enumerate(qa_names, start=1):
                dst_qa.set_band_description(i, qname)

        tmp_stack.replace(out_stack_path)
        tmp_qa.replace(out_qa_path)

    return {"stack": out_stack_path, "qa": out_qa_path}


def build_local_biseasonal_stack(
    region: str,
    cfg: Dict[str, Any],
    session=None,
) -> Dict[str, Any]:
    """Ejecuta el pipeline local completo S2 (STAC) + S1 (ASF) -> Stack 24b + QA 6b en SEPAL."""
    region_key = region.upper()
    reg_cfg = cfg["regiones"][region_key]
    stack_cfg = cfg["stack"]

    aoi_gdf = ensure_local_aoi(region_key, cfg)
    grid_info = compute_region_grid(
        aoi_gdf,
        crs=reg_cfg["crs"],
        scale_m=float(stack_cfg["escala_m"]),
        buffer_m=float(stack_cfg["buffer_stack_m"]),
        margen_kernel_m=float(stack_cfg["margen_kernel_m"]),
    )
    mask_base, mask_buffer = rasterize_aoi_masks(grid_info)

    t_seca = reg_cfg["temporadas"]["SECA"]
    t_lluvia = reg_cfg["temporadas"]["LLUVIA"]
    orbit_pass = reg_cfg["orbit_pass"]

    s2_seca = build_local_s2_composite(region_key, "SECA", grid_info, t_seca["inicio"], t_seca["fin"], cfg)
    s2_lluvia = build_local_s2_composite(region_key, "LLUVIA", grid_info, t_lluvia["inicio"], t_lluvia["fin"], cfg)
    s1_seca = build_local_s1_composite(region_key, "SECA", grid_info, t_seca["inicio"], t_seca["fin"], orbit_pass, cfg, session=session)
    s1_lluvia = build_local_s1_composite(region_key, "LLUVIA", grid_info, t_lluvia["inicio"], t_lluvia["fin"], orbit_pass, cfg, session=session)

    paths = get_local_stack_paths(region_key, cfg)
    bandas_24 = get_predictor_band_names(cfg)["todas"]
    block_rows = int(cfg.get("computo", {}).get("block_size", 1024))

    tags = {
        "version": str(cfg["proyecto"]["version_pipeline"]),
        "region": region_key,
        "fuente_s2": "AWS_Earth_Search_STAC_C1",
        "fuente_s1": str(s1_seca["fuente"]),
        "orbita_s1": str(orbit_pass),
        "s1_speckle": str(cfg["s1"]["speckle_filter"]),
    }

    assemble_local_biseasonal_stack(
        s2_seca["image_path"],
        s2_seca["nobs_path"],
        s2_lluvia["image_path"],
        s2_lluvia["nobs_path"],
        s1_seca["image_path"],
        s1_seca["nobs_path"],
        s1_lluvia["image_path"],
        s1_lluvia["nobs_path"],
        mask_base,
        mask_buffer,
        bandas_24,
        paths["stack"],
        paths["qa"],
        block_rows=block_rows,
        metadata_tags=tags,
    )

    inv_s2 = pd.concat(
        [
            s2_seca["inventario"].assign(temporada="SECA"),
            s2_lluvia["inventario"].assign(temporada="LLUVIA"),
        ],
        ignore_index=True,
    )
    inv_s1 = pd.concat([s1_seca["inventario"], s1_lluvia["inventario"]], ignore_index=True)
    inv_s2.to_csv(paths["inventario_s2"], index=False)
    inv_s1.to_csv(paths["inventario_s1"], index=False)

    return {
        "region": region_key,
        "grid": grid_info,
        "stack_path": paths["stack"],
        "qa_path": paths["qa"],
        "s2_seca": s2_seca,
        "s2_lluvia": s2_lluvia,
        "s1_seca": s1_seca,
        "s1_lluvia": s1_lluvia,
        "inventario_s2": inv_s2,
        "inventario_s1": inv_s1,
    }


def diagnose_local_stack(
    stack_path: Path,
    qa_path: Path,
    target_bands: Sequence[str] = ("VV_SECA", "VH_SECA", "VV_LLUVIA", "VH_LLUVIA", "B4_SECA", "B8_SECA", "B8_LLUVIA"),
    max_sample: int = 50000,
    seed: int = 20250822,
) -> Dict[str, Any]:
    """Ejecuta controles cuantitativos sobre el Stack Biestacional local."""
    rng = np.random.default_rng(seed)

    with rasterio.open(qa_path) as dqa, rasterio.open(stack_path) as dstk:
        desc_stack = list(dstk.descriptions)
        band_idx = {name: i + 1 for i, name in enumerate(desc_stack)}
        height, width = dstk.height, dstk.width

        n_base_total = 0
        n_base_valid = 0
        sample_cols: Dict[str, list[np.ndarray]] = {b: [] for b in target_bands if b in band_idx}

        # Recorrer en franjas para mantener RAM minima
        step_r = 512
        for r0 in range(0, height, step_r):
            h = min(step_r, height - r0)
            win = Window(0, r0, width, h)
            com = dqa.read(5, window=win) == 1
            base = dqa.read(6, window=win) == 1
            valid_base = com & base
            n_base_total += int(base.sum())
            n_base_valid += int(valid_base.sum())

            ys, xs = np.where(valid_base)
            if ys.size > 0:
                k = min(ys.size, max(200, int(max_sample * h / height)))
                sel = rng.choice(ys.size, size=k, replace=False) if ys.size > k else np.arange(ys.size)
                sy, sx = ys[sel], xs[sel]
                for b in sample_cols:
                    arr = dstk.read(band_idx[b], window=win)
                    sample_cols[b].append(arr[sy, sx])

        stats_rows = []
        for b, chunks in sample_cols.items():
            vals = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            stats_rows.append(
                {
                    "banda": b,
                    "n_muestra": int(vals.size),
                    "min": float(np.min(vals)) if vals.size else np.nan,
                    "mean": float(np.mean(vals)) if vals.size else np.nan,
                    "max": float(np.max(vals)) if vals.size else np.nan,
                    "std": float(np.std(vals)) if vals.size else np.nan,
                }
            )

        frac_valida = float(n_base_valid / max(n_base_total, 1))
        size_mb = float(Path(stack_path).stat().st_size / (1024 ** 2))
        return {
            "dimensiones": (height, width),
            "crs": str(dstk.crs),
            "n_bandas_stack": dstk.count,
            "bandas_stack": desc_stack,
            "n_pixeles_aoi_base": n_base_total,
            "n_pixeles_validos_aoi": n_base_valid,
            "fraccion_valida_aoi": frac_valida,
            "tamano_stack_mb": size_mb,
            "stats_df": pd.DataFrame(stats_rows),
        }


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def add_local_raster_overlay(
    map_obj: Any,
    raster_path: Path,
    bands: Sequence[int | str] = (1,),
    layer_name: str = "Capa local",
    stretch: Tuple[float, float] = (0.0, 0.35),
    palette: Optional[Sequence[str] | Dict[int, str]] = None,
    categorical: bool = False,
    nodata_val: float = 0.0,
    max_dim: int = 1200,
    opacity: float = 1.0,
    visible: bool = True,
) -> Any:
    """Superpone un GeoTIFF local sobre un mapa ipyleaflet/geemap sin usar localtileserver.

    Reproyecta un preview ligero en RAM a EPSG:3857 (Web Mercator) y lo incrusta como
    PNG base64 en `ipyleaflet.ImageOverlay`, garantizando compatibilidad 100% detrás
    del proxy HTTPS de SEPAL.io.
    """
    from ipyleaflet import ImageOverlay

    with rasterio.open(raster_path) as src:
        desc = list(src.descriptions)
        idx_list = []
        for b in bands:
            if isinstance(b, int):
                idx_list.append(b)
            elif b in desc:
                idx_list.append(desc.index(b) + 1)
            else:
                raise KeyError(f"Banda '{b}' no encontrada en {raster_path}.")

        # Calcular bounds del raster completo en Web Mercator para evitar distorsión sobre basemap
        src_nodata = src.nodata
        left_4326, bottom_4326, right_4326, top_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        left_m, bottom_m, right_m, top_m = transform_bounds("EPSG:4326", "EPSG:3857", left_4326, bottom_4326, right_4326, top_4326)
        aspect = (right_m - left_m) / max(top_m - bottom_m, 1e-6)
        if aspect >= 1.0:
            out_w = max_dim
            out_h = max(1, int(round(max_dim / aspect)))
        else:
            out_h = max_dim
            out_w = max(1, int(round(max_dim * aspect)))

        merc_transform = rasterio.transform.from_bounds(left_m, bottom_m, right_m, top_m, out_w, out_h)
        resamp = Resampling.nearest if categorical else Resampling.bilinear

        with WarpedVRT(
            src,
            crs="EPSG:3857",
            transform=merc_transform,
            width=out_w,
            height=out_h,
            resampling=resamp,
        ) as vrt:
            arr = vrt.read(idx_list)

    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    if categorical:
        cat = arr[0]
        color_map: Dict[int, Tuple[int, int, int]] = {}
        if isinstance(palette, dict):
            color_map = {int(k): _hex_to_rgb(v) for k, v in palette.items()}
        elif isinstance(palette, (list, tuple)):
            for i, col in enumerate(palette, start=1):
                color_map[i] = _hex_to_rgb(col)
        for val, rgb in color_map.items():
            m = cat == val
            rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = rgb[0], rgb[1], rgb[2], int(round(255 * opacity))
    elif len(idx_list) == 3:
        vmin, vmax = float(stretch[0]), float(stretch[1])
        valid = np.all(np.isfinite(arr), axis=0)
        if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            valid &= np.all(arr != src_nodata, axis=0)
        for ch in range(3):
            scaled = np.clip((arr[ch] - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
            rgba[:, :, ch] = np.where(valid, (scaled * 255.0).astype(np.uint8), 0)
        rgba[:, :, 3] = np.where(valid, int(round(255 * opacity)), 0)
    else:
        band = arr[0]
        valid = np.isfinite(band) & (band != nodata_val if nodata_val is not None else True)
        vmin, vmax = float(stretch[0]), float(stretch[1])
        norm = np.clip((band - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
        if palette and isinstance(palette, (list, tuple)) and len(palette) >= 2:
            ramp = np.array([_hex_to_rgb(c) for c in palette], dtype=np.float32)
            pos = norm * (len(ramp) - 1)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, len(ramp) - 1)
            frac = (pos - lo)[:, :, None]
            rgb_img = (1.0 - frac) * ramp[lo] + frac * ramp[hi]
            rgba[:, :, :3] = np.where(valid[:, :, None], np.clip(rgb_img, 0, 255).astype(np.uint8), 0)
        else:
            g = np.where(valid, (norm * 255.0).astype(np.uint8), 0)
            rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = g
        rgba[:, :, 3] = np.where(valid, int(round(255 * opacity)), 0)

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    overlay = ImageOverlay(
        url=data_url,
        bounds=((bottom_4326, left_4326), (top_4326, right_4326)),
        name=layer_name,
        opacity=float(opacity),
    )
    if hasattr(overlay, "visible"):
        overlay.visible = bool(visible)
    map_obj.add_layer(overlay)
    return overlay
