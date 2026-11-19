"""Preprocesamiento óptico Sentinel-2 L2A (SR Harmonized) en Python GEE.

Soporta dos métodos de enmascaramiento de nubes y sombras:
1. ``cloud_score_plus`` (recomendado): usa ``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED``
   (Pasquarella et al., 2023), más robusto sobre bosques costeros y canales estuarinos
   oscuros que los umbrales geométricos de sombra tradicionales.
2. ``s2cloudless``: usa ``COPERNICUS/S2_CLOUD_PROBABILITY`` + proyección geométrica de
   sombras con protección de agua/manglar inundado mediante SCL y MNDWI.
"""

from __future__ import annotations

from typing import Any, Dict

import ee


def _mask_cloud_score_plus(
    image: ee.Image,
    bandas: list[str],
    cs_band: str,
    cs_threshold: float,
) -> ee.Image:
    """Enmascara nubes/sombras con Cloud Score+ y píxeles saturados/cirros de SCL."""
    clear = image.select(cs_band).gte(cs_threshold)
    scl = image.select("SCL")
    # SCL 1 = saturado/defectuoso, SCL 10 = cirro delgado
    scl_valido = scl.neq(1).And(scl.neq(10))
    mascara_final = clear.And(scl_valido)

    return ee.Image(
        image.select(bandas)
        .divide(10000.0)
        .updateMask(mascara_final)
        .resample("bilinear")
        .copyProperties(
            image,
            ["system:time_start", "system:index", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE"],
        )
    )


def _mask_s2cloudless(
    image: ee.Image,
    cfg_s2: Dict[str, Any],
) -> ee.Image:
    """Enmascara nubes y sombras con s2cloudless + protección MNDWI/SCL de manglar inundado."""
    bandas = list(cfg_s2["bandas"])
    cloud_prob = ee.Image(image.get("cloud_probability")).select("probability")
    clouds = cloud_prob.gt(cfg_s2["probabilidad_nube"]).rename("clouds")

    scl = image.select("SCL")
    scl_valido = scl.neq(1).And(scl.neq(10))

    # Protege canales mareales y manglar inundado de ser confundidos con sombra
    mndwi = image.normalizedDifference(["B3", "B11"])
    not_water = scl.neq(6).And(mndwi.lt(cfg_s2["umbral_mndwi_agua"]))

    dark_pixels = (
        image.select("B8")
        .lt(cfg_s2["reflectancia_nir_oscura"] * 1e4)
        .And(not_water)
        .rename("dark_pixels")
    )

    shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    cloud_transform = (
        clouds.directionalDistanceTransform(shadow_azimuth, cfg_s2["distancia_sombra_km"] * 10)
        .reproject(crs=image.select("B2").projection(), scale=100)
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    shadows = cloud_transform.And(dark_pixels).rename("shadows")
    cloud_or_shadow = (
        clouds.Or(shadows)
        .focalMin(radius=cfg_s2["erosion_m"], units="meters")
        .focalMax(radius=cfg_s2["buffer_m"], units="meters")
    )

    mascara_final = cloud_or_shadow.Not().And(scl_valido)

    return ee.Image(
        image.select(bandas)
        .divide(10000.0)
        .updateMask(mascara_final)
        .resample("bilinear")
        .copyProperties(
            image,
            ["system:time_start", "system:index", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE"],
        )
    )


def build_s2_seasonal_composite(
    aoi_proc: ee.Geometry,
    aoi_clip: ee.Geometry,
    inicio: str,
    fin: str,
    cfg_s2: Dict[str, Any],
    crs: str,
    crs_transform: list[float],
    temporada: str,
) -> Dict[str, Any]:
    """Construye el mosaico mediano estacional de Sentinel-2 SR armonizado a 10 m.

    Parameters
    ----------
    aoi_proc : ee.Geometry
        Geometría extendida (AOI + buffer stack + margen de kernel).
    aoi_clip : ee.Geometry
        Geometría de recorte final (AOI + buffer stack).
    inicio, fin : str
        Fechas ISO ('YYYY-MM-DD'); ``fin`` es exclusiva en GEE ``filterDate``.
    cfg_s2 : dict
        Sección ``s2`` de ``config.yaml``.
    crs : str
        CRS métrico objetivo (ej. ``EPSG:32615`` o ``EPSG:32616``).
    crs_transform : list[float]
        Transformación afín exacta a 10 m.
    temporada : str
        Etiqueta de temporada (``'SECA'`` o ``'LLUVIA'``).
    """
    bandas = list(cfg_s2["bandas"])
    metodo = str(cfg_s2.get("metodo_nubes", "cloud_score_plus")).lower()
    bandas_salida = [f"{b}_{temporada}" for b in bandas]

    sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_proc)
        .filterDate(inicio, fin)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cfg_s2["nubosidad_escena_max"]))
    )

    if metodo == "cloud_score_plus":
        cs_band = cfg_s2.get("cs_band", "cs_cdf")
        cs_threshold = float(cfg_s2.get("cs_threshold", 0.60))
        cs_plus = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filterBounds(aoi_proc)
            .filterDate(inicio, fin)
        )
        joined = sr.linkCollection(cs_plus, [cs_band])
        validas = joined.map(
            lambda img: _mask_cloud_score_plus(ee.Image(img), bandas, cs_band, cs_threshold)
        )
    elif metodo == "s2cloudless":
        prob = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(aoi_proc)
            .filterDate(inicio, fin)
        )
        joined = ee.ImageCollection(
            ee.Join.saveFirst("cloud_probability").apply(
                primary=sr,
                secondary=prob,
                condition=ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            )
        ).filter(ee.Filter.notNull(["cloud_probability"]))
        validas = joined.map(lambda img: _mask_s2cloudless(ee.Image(img), cfg_s2))
    else:
        raise ValueError(
            f"Método de nubes S2 no reconocido: '{metodo}'. "
            "Usa 'cloud_score_plus' o 's2cloudless'."
        )

    composite = (
        validas.select(bandas)
        .median()
        .setDefaultProjection(crs=crs, crsTransform=crs_transform)
        .rename(bandas_salida)
        .clip(aoi_clip)
        .toFloat()
    )

    nobs = (
        validas.select("B2")
        .count()
        .setDefaultProjection(crs=crs, crsTransform=crs_transform)
        .rename(f"NOBS_S2_{temporada}")
        .clip(aoi_clip)
        .toUint16()
    )

    return {
        "image": composite,
        "nobs": nobs,
        "count": joined.size(),
        "ids": joined.aggregate_array("system:index"),
        "tiles": joined.aggregate_array("MGRS_TILE").distinct().sort(),
        "metodo_nubes": metodo,
    }
