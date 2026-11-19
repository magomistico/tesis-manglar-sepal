"""Ensamblaje, diagnóstico y exportación del Stack Biestacional S1 + S2 (24 bandas + 6 QA).

Construye desde cero el stack armonizado a 10 m con doble anillo geométrico
(buffer externo de 100 m + margen interno de 60 m para kernels) para las costas
del Pacífico (EPSG:32615) y Caribe (EPSG:32616) de Guatemala.
"""

from __future__ import annotations

from typing import Any, Dict

import ee

from .config_utils import get_predictor_band_names, load_aoi_geometry
from .s1_ard import build_s1_seasonal_composite, inspect_s1_orbits
from .s2_preproc import build_s2_seasonal_composite


def build_biseasonal_stack(
    region: str,
    cfg: Dict[str, Any],
    aoi_geometry: ee.Geometry | None = None,
) -> Dict[str, Any]:
    """Construye el Stack Biestacional de 24 bandas y la imagen QA de 6 bandas.

    Parameters
    ----------
    region : str
        ``'PACIFICO'`` o ``'CARIBE'``.
    cfg : dict
        Configuración cargada desde ``config.yaml``.
    aoi_geometry : ee.Geometry, optional
        Geometría AOI explícita. Si es ``None``, se lee del asset en ``config.yaml``.
    """
    region_key = region.upper()
    if region_key not in cfg["regiones"]:
        raise ValueError(f"Región desconocida: '{region}'. Usa 'PACIFICO' o 'CARIBE'.")

    reg_cfg = cfg["regiones"][region_key]
    stack_cfg = cfg["stack"]
    crs = reg_cfg["crs"]
    crs_transform = list(reg_cfg["crs_transform"])
    version = cfg["proyecto"]["version_pipeline"]
    max_pixels = int(cfg["gee"]["max_pixels"])

    if aoi_geometry is None:
        aoi_asset = cfg["assets"]["aoi"][region_key]
        aoi_base = load_aoi_geometry(aoi_asset)
    else:
        aoi_base = ee.Geometry(aoi_geometry)

    # Doble anillo geométrico en metros geodésicos (evita efectos de borde en filtros focales)
    buffer_m = float(stack_cfg["buffer_stack_m"])
    margen_m = float(stack_cfg["margen_kernel_m"])
    aoi_buffer = aoi_base.buffer(buffer_m, 1)
    aoi_proc = aoi_buffer.buffer(margen_m, 1)

    t_seca = reg_cfg["temporadas"]["SECA"]
    t_lluvia = reg_cfg["temporadas"]["LLUVIA"]
    orbit_pass = reg_cfg["orbit_pass"]

    # Rama óptica Sentinel-2
    s2_seca = build_s2_seasonal_composite(
        aoi_proc, aoi_buffer, t_seca["inicio"], t_seca["fin"],
        cfg["s2"], crs, crs_transform, "SECA",
    )
    s2_lluvia = build_s2_seasonal_composite(
        aoi_proc, aoi_buffer, t_lluvia["inicio"], t_lluvia["fin"],
        cfg["s2"], crs, crs_transform, "LLUVIA",
    )

    # Rama radar Sentinel-1 ARD
    s1_seca = build_s1_seasonal_composite(
        aoi_proc, aoi_buffer, t_seca["inicio"], t_seca["fin"],
        cfg["s1"], orbit_pass, crs, crs_transform, "SECA",
        region_key, version, max_pixels,
    )
    s1_lluvia = build_s1_seasonal_composite(
        aoi_proc, aoi_buffer, t_lluvia["inicio"], t_lluvia["fin"],
        cfg["s1"], orbit_pass, crs, crs_transform, "LLUVIA",
        region_key, version, max_pixels,
    )

    band_groups = get_predictor_band_names(cfg)
    bandas_24 = band_groups["todas"]

    stack_raw = (
        s2_seca["image"]
        .addBands(s2_lluvia["image"])
        .addBands(s1_seca["image"])
        .addBands(s1_lluvia["image"])
        .select(bandas_24)
    )

    # Máscara común estricta: solo píxeles válidos en las 24 bandas
    common_mask = stack_raw.mask().reduce(ee.Reducer.min())

    stack = stack_raw.updateMask(common_mask).set(
        {
            "version": version,
            "region": region_key,
            "producto": "STACK_BIESTACIONAL_S1_S2",
            "buffer_aoi_m": buffer_m,
            "margen_kernel_m": margen_m,
            "fecha_seca_inicio": t_seca["inicio"],
            "fecha_seca_fin_exclusiva": t_seca["fin"],
            "fecha_lluvia_inicio": t_lluvia["inicio"],
            "fecha_lluvia_fin_exclusiva": t_lluvia["fin"],
            "s2_metodo_nubes": s2_seca["metodo_nubes"],
            "s1_orbita": orbit_pass,
            "s1_speckle": cfg["s1"]["speckle_filter"],
            "s1_terrain_model": cfg["s1"]["terrain_model"],
            "bandas_predictoras": len(bandas_24),
            "unidad_sar": "dB_gamma0",
            "unidad_s2": "reflectancia_superficie_0_1",
            "crs_objetivo": crs,
            "crs_transform": ",".join(str(x) for x in crs_transform),
        }
    )

    # Banda indicadora: 1 = dentro del AOI oficial, 0 = anillo buffer de 100 m
    mascara_aoi_base = (
        ee.Image(0)
        .byte()
        .paint(ee.FeatureCollection([ee.Feature(aoi_base)]), 1)
        .setDefaultProjection(crs=crs, crsTransform=crs_transform)
        .clip(aoi_buffer)
        .rename("MASCARA_AOI_BASE")
        .toUint16()
    )

    qa = ee.Image.cat(
        [
            s2_seca["nobs"],
            s2_lluvia["nobs"],
            s1_seca["nobs"],
            s1_lluvia["nobs"],
            common_mask.rename("MASCARA_COMUN").toUint16(),
            mascara_aoi_base,
        ]
    ).set(
        {
            "version": version,
            "region": region_key,
            "buffer_aoi_m": buffer_m,
            "uso": "CONTROL_CALIDAD_NO_PREDICTOR",
        }
    )

    inventory_s1 = s1_seca["inventory"].merge(s1_lluvia["inventory"])
    def _join_num_list(ee_list: ee.List) -> ee.String:
        return ee.List(ee_list).map(lambda n: ee.Number(n).format("%d")).join(", ")

    metadata_fc = ee.FeatureCollection(
        [
            ee.Feature(
                None,
                {
                    "version": version,
                    "region": region_key,
                    "temporada": "SECA",
                    "buffer_m": buffer_m,
                    "inicio": t_seca["inicio"],
                    "fin_exclusivo": t_seca["fin"],
                    "metodo_nubes_s2": s2_seca["metodo_nubes"],
                    "n_s2": s2_seca["count"],
                    "ids_s2": s2_seca["ids"].join(", "),
                    "tiles_s2": s2_seca["tiles"].join(", "),
                    "orbita_s1": orbit_pass,
                    "n_s1_raw": s1_seca["count_raw"],
                    "n_s1_procesadas": s1_seca["count_processed"],
                    "orbitas_relativas_s1": _join_num_list(s1_seca["relative_orbits"]),
                },
            ),
            ee.Feature(
                None,
                {
                    "version": version,
                    "region": region_key,
                    "temporada": "LLUVIA",
                    "buffer_m": buffer_m,
                    "inicio": t_lluvia["inicio"],
                    "fin_exclusivo": t_lluvia["fin"],
                    "metodo_nubes_s2": s2_lluvia["metodo_nubes"],
                    "n_s2": s2_lluvia["count"],
                    "ids_s2": s2_lluvia["ids"].join(", "),
                    "tiles_s2": s2_lluvia["tiles"].join(", "),
                    "orbita_s1": orbit_pass,
                    "n_s1_raw": s1_lluvia["count_raw"],
                    "n_s1_procesadas": s1_lluvia["count_processed"],
                    "orbitas_relativas_s1": _join_num_list(s1_lluvia["relative_orbits"]),
                },
            ),
        ]
    )

    return {
        "region": region_key,
        "crs": crs,
        "crs_transform": crs_transform,
        "aoi_base": aoi_base,
        "aoi_buffer": aoi_buffer,
        "aoi_proc": aoi_proc,
        "stack": stack,
        "qa": qa,
        "s2_seca": s2_seca,
        "s2_lluvia": s2_lluvia,
        "s1_seca": s1_seca,
        "s1_lluvia": s1_lluvia,
        "inventory_s1": inventory_s1,
        "metadata_fc": metadata_fc,
    }


def diagnose_stack(bundle: Dict[str, Any], cfg: Dict[str, Any], n_pixels: int = 80) -> Dict[str, Any]:
    """Ejecuta controles cuantitativos rápidos sobre el stack antes de exportar."""
    stack = bundle["stack"]
    qa = bundle["qa"]
    aoi_base = bundle["aoi_base"]
    seed = int(cfg["proyecto"]["seed"])

    # Muestra aleatoria rápida para verificar rangos físicos (dB negativos en SAR, [0, 1] en S2)
    sample = stack.select(
        ["VV_SECA", "VH_SECA", "VV_LLUVIA", "VH_LLUVIA", "B4_SECA", "B8_SECA", "B8_LLUVIA"]
    ).sample(
        region=aoi_base,
        scale=30,
        numPixels=n_pixels,
        seed=seed,
        tileScale=8,
        geometries=False,
    )

    # Cobertura válida dentro del AOI base
    valid_frac = (
        qa.select("MASCARA_COMUN")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_base,
            scale=30,
            bestEffort=True,
            maxPixels=int(cfg["gee"]["max_pixels"]),
        )
        .get("MASCARA_COMUN")
    )

    diag = ee.Dictionary(
        {
            "bandas_stack": stack.bandNames(),
            "bandas_qa": qa.bandNames(),
            "n_s2_seca": bundle["s2_seca"]["count"],
            "n_s2_lluvia": bundle["s2_lluvia"]["count"],
            "tiles_s2_seca": bundle["s2_seca"]["tiles"],
            "tiles_s2_lluvia": bundle["s2_lluvia"]["tiles"],
            "n_s1_seca": bundle["s1_seca"]["count_processed"],
            "n_s1_lluvia": bundle["s1_lluvia"]["count_processed"],
            "orbitas_rel_s1_seca": bundle["s1_seca"]["relative_orbits"],
            "orbitas_rel_s1_lluvia": bundle["s1_lluvia"]["relative_orbits"],
            "fraccion_valida_aoi": valid_frac,
            "n_muestra_control": sample.size(),
            "stats_VV_SECA": sample.aggregate_stats("VV_SECA"),
            "stats_VH_SECA": sample.aggregate_stats("VH_SECA"),
            "stats_VV_LLUVIA": sample.aggregate_stats("VV_LLUVIA"),
            "stats_VH_LLUVIA": sample.aggregate_stats("VH_LLUVIA"),
            "stats_B8_SECA": sample.aggregate_stats("B8_SECA"),
            "stats_B8_LLUVIA": sample.aggregate_stats("B8_LLUVIA"),
        }
    ).getInfo()

    return diag


def get_stack_asset_ids(region: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    """Devuelve los Asset IDs estandarizados para stack, QA y metadatos de una región."""
    reg = region.lower()
    ver = cfg["proyecto"]["version_pipeline"].lower()
    root = cfg["gee"]["asset_root"].rstrip("/") + "/"
    return {
        "stack": f"{root}stack_biestacional_{reg}_2025_{ver}",
        "qa": f"{root}qa_stack_biestacional_{reg}_2025_{ver}",
        "inventario_s1": f"{root}inventario_s1_{reg}_2025_{ver}",
        "metadatos": f"{root}metadatos_stack_{reg}_2025_{ver}",
    }


def start_stack_exports(
    bundle: Dict[str, Any],
    cfg: Dict[str, Any],
    export_tables_to_drive: bool = True,
) -> Dict[str, Any]:
    """Lanza las tareas de exportación del stack, QA e inventarios a GEE Assets y Drive."""
    region = bundle["region"]
    ver = cfg["proyecto"]["version_pipeline"]
    ids = get_stack_asset_ids(region, cfg)
    drive_folder = cfg["gee"]["drive_folder"]
    max_pixels = int(cfg["gee"]["max_pixels"])

    tasks: Dict[str, Any] = {}

    t_stack = ee.batch.Export.image.toAsset(
        image=bundle["stack"],
        description=f"Stack_Biestacional_{region}_2025_{ver}",
        assetId=ids["stack"],
        region=bundle["aoi_buffer"],
        crs=bundle["crs"],
        crsTransform=bundle["crs_transform"],
        maxPixels=max_pixels,
        pyramidingPolicy={".default": "mean"},
    )
    t_stack.start()
    tasks["stack_asset"] = {"task_id": t_stack.id, "asset_id": ids["stack"]}

    t_qa = ee.batch.Export.image.toAsset(
        image=bundle["qa"],
        description=f"QA_Stack_{region}_2025_{ver}",
        assetId=ids["qa"],
        region=bundle["aoi_buffer"],
        crs=bundle["crs"],
        crsTransform=bundle["crs_transform"],
        maxPixels=max_pixels,
        pyramidingPolicy={".default": "mode"},
    )
    t_qa.start()
    tasks["qa_asset"] = {"task_id": t_qa.id, "asset_id": ids["qa"]}

    t_inv_asset = ee.batch.Export.table.toAsset(
        collection=bundle["inventory_s1"],
        description=f"Asset_Inventario_S1_{region}_{ver}",
        assetId=ids["inventario_s1"],
    )
    t_inv_asset.start()
    tasks["inventario_s1_asset"] = {"task_id": t_inv_asset.id, "asset_id": ids["inventario_s1"]}

    t_meta_asset = ee.batch.Export.table.toAsset(
        collection=bundle["metadata_fc"],
        description=f"Asset_Metadatos_{region}_{ver}",
        assetId=ids["metadatos"],
    )
    t_meta_asset.start()
    tasks["metadatos_asset"] = {"task_id": t_meta_asset.id, "asset_id": ids["metadatos"]}

    if export_tables_to_drive:
        t_inv_drv = ee.batch.Export.table.toDrive(
            collection=bundle["inventory_s1"],
            description=f"Inventario_S1_{region}_2025_{ver}",
            folder=drive_folder,
            fileFormat="CSV",
        )
        t_inv_drv.start()
        tasks["inventario_s1_drive"] = {"task_id": t_inv_drv.id, "folder": drive_folder}

        t_meta_drv = ee.batch.Export.table.toDrive(
            collection=bundle["metadata_fc"],
            description=f"Metadatos_Stack_{region}_2025_{ver}",
            folder=drive_folder,
            fileFormat="CSV",
        )
        t_meta_drv.start()
        tasks["metadatos_drive"] = {"task_id": t_meta_drv.id, "folder": drive_folder}

    return tasks


def check_region_orbits(region: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnostica disponibilidad de órbitas S1 (ASCENDING/DESCENDING) para ambas épocas."""
    region_key = region.upper()
    reg_cfg = cfg["regiones"][region_key]
    aoi_base = load_aoi_geometry(cfg["assets"]["aoi"][region_key])
    aoi_proc = aoi_base.buffer(
        float(cfg["stack"]["buffer_stack_m"]) + float(cfg["stack"]["margen_kernel_m"]), 1
    )
    return {
        "SECA": inspect_s1_orbits(
            aoi_proc, reg_cfg["temporadas"]["SECA"]["inicio"], reg_cfg["temporadas"]["SECA"]["fin"]
        ),
        "LLUVIA": inspect_s1_orbits(
            aoi_proc, reg_cfg["temporadas"]["LLUVIA"]["inicio"], reg_cfg["temporadas"]["LLUVIA"]["fin"]
        ),
    }
