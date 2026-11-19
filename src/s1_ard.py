"""Preprocesamiento Sentinel-1 GRD ARD en Python puro para Google Earth Engine.

Implementa de forma autocontenida (sin depender de `require()` de JS) la cadena
Analysis-Ready Data (ARD) para Sentinel-1 IW dual-pol (VV, VH):
1. Filtrado de colección `COPERNICUS/S1_GRD_FLOAT` (escala lineal, ya corregida en
   órbita, ruido térmico, calibración radiométrica sigma0 y Range-Doppler en GEE).
2. Corrección adicional de ruido de borde por ángulo de incidencia (Hird et al., 2017;
   Mullissa et al., 2021).
3. Filtrado de moteado (speckle) en potencia lineal: Refined Lee 7x7 direccional
   preservador de bordes (Lee, 1981; Lemoine; Mullissa et al., 2021) o Boxcar.
4. Aplanamiento radiométrico del terreno (sigma0 -> gamma0_T) por modelo angular de
   dispersión volumétrica o superficial (Vollrath, Mullissa & Reiche, 2020) usando
   DEM costero con `.unmask(0)` para no perder manglares ni esteros a nivel del mar.
5. Conversión segura a decibelios (dB) con piso lineal (1e-6 = -60 dB) y composición
   mediana estacional alineada a la grilla UTM de 10 m.

Referencias:
- Mullissa, A., et al. (2021). Sentinel-1 SAR Backscatter Analysis Ready Data
  Preparation in Google Earth Engine. Remote Sensing, 13(10), 1954.
- Vollrath, A., Mullissa, A., & Reiche, J. (2020). Angular-Based Radiometric Slope
  Correction for Sentinel-1 on Google Earth Engine. Remote Sensing, 12(11), 1867.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import ee


def filter_s1_collection(
    aoi_proc: ee.Geometry,
    inicio: str,
    fin: str,
    orbit_pass: str,
) -> ee.ImageCollection:
    """Filtra Sentinel-1 GRD FLOAT en modo IW, dual-pol VV+VH y órbita consistente."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
        .filterBounds(aoi_proc)
        .filterDate(inicio, fin)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .select(["VV", "VH", "angle"])
    )


def inspect_s1_orbits(
    aoi_proc: ee.Geometry,
    inicio: str,
    fin: str,
) -> Dict[str, Dict[str, Any]]:
    """Diagnostica disponibilidad de escenas ASCENDING vs DESCENDING para un AOI y periodo."""
    resumen: Dict[str, Dict[str, Any]] = {}
    for orbit in ("ASCENDING", "DESCENDING"):
        col = filter_s1_collection(aoi_proc, inicio, fin, orbit)
        resumen[orbit] = {
            "n_escenas": col.size().getInfo(),
            "orbitas_relativas": col.aggregate_array("relativeOrbitNumber_start")
            .distinct()
            .sort()
            .getInfo(),
        }
    return resumen


def border_noise_correction(image: ee.Image) -> ee.Image:
    """Enmascara bordes con ángulos de incidencia extremos (30.64° < angle < 45.24°)."""
    angle = image.select("angle")
    valid_angle = angle.gt(30.63993).And(angle.lt(45.23993))
    # También descartar píxeles lineales no positivos
    valid_power = image.select("VV").gt(0).And(image.select("VH").gt(0))
    return ee.Image(
        image.updateMask(valid_angle.And(valid_power)).copyProperties(
            image, image.propertyNames()
        )
    )


def _refined_lee_single_band(img: ee.Image, band_name: str) -> ee.Image:
    """Aplica filtro Refined Lee 7x7 direccional sobre una banda en potencia lineal."""
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    var3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # Muestrea las 9 sub-ventanas 3x3 dentro de la ventana 7x7
    sample_weights = ee.List(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = var3.neighborhoodToBands(sample_kernel)

    # 4 gradientes direccionales
    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(
        sample_mean.select(6).subtract(sample_mean.select(2)).abs()
    )
    gradients = gradients.addBands(
        sample_mean.select(3).subtract(sample_mean.select(5)).abs()
    )
    gradients = gradients.addBands(
        sample_mean.select(0).subtract(sample_mean.select(8)).abs()
    )

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    # 8 direcciones (1..8)
    directions = (
        sample_mean.select(1)
        .subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(7)))
        .multiply(1)
    )
    directions = directions.addBands(
        sample_mean.select(6)
        .subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(2)))
        .multiply(2)
    )
    directions = directions.addBands(
        sample_mean.select(3)
        .subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(5)))
        .multiply(3)
    )
    directions = directions.addBands(
        sample_mean.select(0)
        .subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(8)))
        .multiply(4)
    )
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    directions = (
        directions.updateMask(gradmask)
        .selfMask()
        .reduce(ee.Reducer.firstNonNull())
    )

    # Varianza normalizada de ruido local (media de las 5 sub-ventanas más homogéneas)
    # Se aplana a imagen escalar aquí para que .clamp(0, 1) opere sobre escalares en GEE
    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean).max(1e-12))
    sigma_v = (
        sample_stats.toArray()
        .arraySort()
        .arraySlice(0, 0, 5)
        .arrayReduce(ee.Reducer.mean(), [0])
        .arrayProject([0])
        .arrayFlatten([["sigma_v"]])
    )

    # Kernels direccionales 7x7 (rectángulo + diagonal y sus 3 rotaciones)
    rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(
        ee.List.repeat(ee.List.repeat(1, 7), 4)
    )
    diag_weights = ee.List(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(
        directions.eq(1)
    )
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(
        directions.eq(1)
    )

    dir_mean = dir_mean.addBands(
        img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(
            directions.eq(2)
        )
    )
    dir_var = dir_var.addBands(
        img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(
            directions.eq(2)
        )
    )

    for rot in range(1, 4):
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(rot)).updateMask(
                directions.eq(2 * rot + 1)
            )
        )
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(
                ee.Reducer.variance(), rect_kernel.rotate(rot)
            ).updateMask(directions.eq(2 * rot + 1))
        )
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(rot)).updateMask(
                directions.eq(2 * rot + 2)
            )
        )
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(
                ee.Reducer.variance(), diag_kernel.rotate(rot)
            ).updateMask(directions.eq(2 * rot + 2))
        )

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    # Estimador MMSE con recorte de peso b en [0, 1] para estabilidad sobre agua calma
    var_x = (
        dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v))
        .divide(sigma_v.add(1.0))
    )
    b_weight = var_x.divide(dir_var.max(1e-12)).clamp(0.0, 1.0)
    result = dir_mean.add(b_weight.multiply(img.subtract(dir_mean)))

    return result.rename(band_name).toFloat()


def apply_speckle_filter(
    image: ee.Image,
    method: str = "REFINED_LEE",
    kernel_size: int = 7,
) -> ee.Image:
    """Aplica filtro de moteado en potencia lineal preservando la banda `angle`."""
    method_norm = method.upper().replace(" ", "_")
    if method_norm == "NONE":
        return image

    if method_norm == "REFINED_LEE":
        vv_f = _refined_lee_single_band(image.select("VV"), "VV")
        vh_f = _refined_lee_single_band(image.select("VH"), "VH")
        filtered = ee.Image.cat([vv_f, vh_f, image.select("angle")])
    elif method_norm == "BOXCAR":
        radius = max(1, int((kernel_size - 1) // 2))
        kernel = ee.Kernel.square(radius, "pixels", True)
        pols = image.select(["VV", "VH"]).convolve(kernel).toFloat()
        filtered = pols.addBands(image.select("angle"))
    else:
        raise ValueError(
            f"Filtro speckle no soportado: '{method}'. Usa 'REFINED_LEE', 'BOXCAR' o 'NONE'."
        )

    return ee.Image(filtered.copyProperties(image, image.propertyNames()))


def _load_coastal_dem(dem_asset: str) -> ee.Image:
    """Carga el DEM y rellena áreas marinas/esteros con 0 m para no perforar la costa."""
    if "COPERNICUS/DEM" in dem_asset:
        return ee.ImageCollection(dem_asset).select("DEM").mosaic().unmask(0)
    return ee.Image(dem_asset).unmask(0)


def radiometric_terrain_flattening(
    collection: ee.ImageCollection,
    dem_asset: str = "USGS/SRTMGL1_003",
    model: str = "VOLUME",
    layover_shadow_buffer_m: float = 0,
) -> ee.ImageCollection:
    """Normalización radiométrica angular del terreno (Vollrath et al., 2020).

    Convierte sigma0 (lineal) a gamma0 aplanado (lineal) usando modelo ``VOLUME``
    (vegetación/manglar, Hoekman & Reiche 2015) o ``DIRECT`` (superficies desnudas).
    """
    dem = _load_coastal_dem(dem_asset)
    ninety_rad = ee.Image.constant(90.0 * math.pi / 180.0)
    model_upper = model.upper()
    if model_upper not in ("VOLUME", "DIRECT"):
        raise ValueError(f"Modelo de terreno inválido: '{model}'. Usa 'VOLUME' o 'DIRECT'.")

    def _erode_mask(mask_img: ee.Image, distance_m: float) -> ee.Image:
        dist = (
            mask_img.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )
        return mask_img.updateMask(dist.gt(distance_m))

    def _correct(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        geom = image.geometry()
        proj = image.select("VV").projection()

        elevation = dem.resample("bilinear").reproject(proj, None, 10).clip(geom)

        heading_dict = ee.Terrain.aspect(image.select("angle")).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            bestEffort=True,
        )
        heading_val = ee.Dictionary(heading_dict).combine({"aspect": 0}, False).get("aspect")
        heading_signed = ee.Algorithms.If(
            ee.Number(heading_val).gt(180),
            ee.Number(heading_val).subtract(360),
            ee.Number(heading_val),
        )

        # Geometría radar
        theta_i = image.select("angle").multiply(math.pi / 180.0)
        phi_i = ee.Image.constant(heading_signed).multiply(math.pi / 180.0)

        # Geometría del terreno (convención Vollrath et al., 2020)
        alpha_s = ee.Terrain.slope(elevation).select("slope").multiply(math.pi / 180.0)
        aspect = ee.Terrain.aspect(elevation).select("aspect").clip(geom)
        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)
        phi_s = (
            aspect.updateMask(aspect.lte(180))
            .unmask(0)
            .add(aspect_minus.unmask(0))
            .multiply(-1.0)
            .multiply(math.pi / 180.0)
        )

        phi_r = phi_i.subtract(phi_s)
        alpha_r = alpha_s.tan().multiply(phi_r.cos()).atan()
        alpha_az = alpha_s.tan().multiply(phi_r.sin()).atan()

        # Sigma0 -> Gamma0 elipsoidal
        gamma0 = image.select(["VV", "VH"]).divide(theta_i.cos())

        if model_upper == "VOLUME":
            scf = ninety_rad.subtract(theta_i).add(alpha_r).tan().divide(
                ninety_rad.subtract(theta_i).tan()
            )
        else:
            scf = ninety_rad.subtract(theta_i).cos().divide(
                alpha_az.cos().multiply(ninety_rad.subtract(theta_i).add(alpha_r).cos())
            )

        gamma0_flat = gamma0.multiply(scf)

        # Máscara activa de layover y shadow
        layover = alpha_r.lt(theta_i)
        shadow = alpha_r.gt(ninety_rad.subtract(theta_i).multiply(-1.0))
        ls_mask = layover.And(shadow)
        if layover_shadow_buffer_m > 0:
            ls_mask = _erode_mask(ls_mask, layover_shadow_buffer_m)

        out = (
            gamma0_flat.updateMask(ls_mask)
            .addBands(image.select("angle"))
            .copyProperties(image, image.propertyNames())
        )
        return ee.Image(out)

    return collection.map(_correct)


def lin_to_db_safe(image: ee.Image, floor: float = 1e-6) -> ee.Image:
    """Convierte VV y VH lineales a dB aplicando piso lineal (evita -Infinity/NaN)."""
    pols = ee.Image(image).select(["VV", "VH"]).max(floor)
    db = pols.log10().multiply(10.0).resample("bilinear")
    return ee.Image(db.copyProperties(image, ["system:time_start", "system:index"]))


def _inventory_scene(
    image: ee.Image,
    aoi_clip: ee.Geometry,
    region: str,
    version: str,
    temporada: str,
    max_pixels: int,
) -> ee.Feature:
    """Registra metadatos de cada escena Sentinel-1 para trazabilidad de tesis."""
    angle_stats = image.select("angle").reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=aoi_clip,
        scale=1000,
        bestEffort=True,
        maxPixels=max_pixels,
    )
    return ee.Feature(
        None,
        {
            "version": version,
            "region": region,
            "temporada": temporada,
            "system_index": image.get("system:index"),
            "fecha_utc": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm"),
            "plataforma": image.get("platform_number"),
            "direccion_orbital": image.get("orbitProperties_pass"),
            "orbita_relativa": image.get("relativeOrbitNumber_start"),
            "angulo_min": angle_stats.get("angle_min"),
            "angulo_max": angle_stats.get("angle_max"),
        },
    )


def build_s1_seasonal_composite(
    aoi_proc: ee.Geometry,
    aoi_clip: ee.Geometry,
    inicio: str,
    fin: str,
    cfg_s1: Dict[str, Any],
    orbit_pass: str,
    crs: str,
    crs_transform: list[float],
    temporada: str,
    region: str,
    version: str,
    max_pixels: int = 10_000_000_000_000,
) -> Dict[str, Any]:
    """Ejecuta el pipeline ARD completo de Sentinel-1 y construye el mosaico mediano en dB."""
    raw = filter_s1_collection(aoi_proc, inicio, fin, orbit_pass)
    col = raw

    if cfg_s1.get("aplicar_border_noise", True):
        col = col.map(border_noise_correction)

    speckle_method = cfg_s1.get("speckle_filter", "REFINED_LEE")
    kernel_size = int(cfg_s1.get("kernel_size", 7))
    if speckle_method and speckle_method.upper() != "NONE":
        col = col.map(lambda img: apply_speckle_filter(ee.Image(img), speckle_method, kernel_size))

    if cfg_s1.get("aplicar_terrain_flattening", True):
        col = radiometric_terrain_flattening(
            col,
            dem_asset=cfg_s1.get("dem_asset", "USGS/SRTMGL1_003"),
            model=cfg_s1.get("terrain_model", "VOLUME"),
            layover_shadow_buffer_m=float(cfg_s1.get("layover_shadow_buffer_m", 0)),
        )

    piso = float(cfg_s1.get("piso_lineal_min", 1e-6))
    col_db = col.map(lambda img: lin_to_db_safe(ee.Image(img), floor=piso))

    names = [f"VV_{temporada}", f"VH_{temporada}"]
    composite = (
        col_db.median()
        .setDefaultProjection(crs=crs, crsTransform=crs_transform)
        .rename(names)
        .clip(aoi_clip)
        .toFloat()
    )

    nobs = (
        col_db.select("VV")
        .count()
        .setDefaultProjection(crs=crs, crsTransform=crs_transform)
        .rename(f"NOBS_S1_{temporada}")
        .clip(aoi_clip)
        .toUint16()
    )

    inventory = ee.FeatureCollection(
        raw.map(
            lambda item: _inventory_scene(
                ee.Image(item), aoi_clip, region, version, temporada, max_pixels
            )
        )
    )

    return {
        "image": composite,
        "nobs": nobs,
        "count_raw": raw.size(),
        "count_processed": col_db.size(),
        "relative_orbits": raw.aggregate_array("relativeOrbitNumber_start").distinct().sort(),
        "inventory": inventory,
    }
