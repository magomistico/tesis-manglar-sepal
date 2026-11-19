"""Utilidades de configuración, rutas e inicialización de Google Earth Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_project_root(start: Optional[Path] = None) -> Path:
    """Encuentra la raíz del repositorio buscando `config.yaml` hacia arriba."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "No se encontró 'config.yaml' en el directorio actual ni en carpetas superiores. "
        "Verifica que estés ejecutando el notebook dentro del repositorio 'tesis-manglar-sepal'."
    )


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Carga `config.yaml` y añade la ruta resuelta `_root`."""
    if config_path is None:
        root = find_project_root()
        path = root / "config.yaml"
    else:
        path = Path(config_path).resolve()
        root = path.parent

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_root"] = str(root)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_output_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Crea y devuelve el diccionario de directorios de salida del proyecto."""
    root = Path(cfg["_root"])
    local_cfg = cfg.get("local_assets", {})
    dirs = {
        "outputs": root / "outputs",
        "aoi": root / local_cfg.get("aoi_dir", "outputs/aoi"),
        "guia": root / local_cfg.get("guia_dir", "outputs/guia"),
        "stack": root / local_cfg.get("stack_dir", "outputs/stack"),
        "tablas": root / "outputs" / "tablas",
        "figuras": root / "outputs" / "figuras",
        "muestras": root / "outputs" / "muestras",
        "modelos": root / "outputs" / "modelos",
        "mapas": root / "outputs" / "mapas",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_max_workers(cfg: Dict[str, Any], cap: int = 16) -> int:
    """Devuelve el número de hilos/procesos recomendados según los CPUs de la VM SEPAL."""
    import os

    configured = cfg.get("computo", {}).get("max_workers")
    if configured is not None:
        return max(1, int(configured))
    cpus = os.cpu_count() or 2
    return max(2, min(cpus, cap))


def get_class_maps(cfg: Dict[str, Any]) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Devuelve diccionarios {id: nombre}, {id: nombre_corto}, {id: color}."""
    clases = cfg["clases"]
    nombres = {int(k): v["nombre"] for k, v in clases.items()}
    cortos = {int(k): v["nombre_corto"] for k, v in clases.items()}
    colores = {int(k): v["color"] for k, v in clases.items()}
    return nombres, cortos, colores


def get_predictor_band_names(cfg: Dict[str, Any]) -> Dict[str, list[str]]:
    """Devuelve las listas ordenadas de nombres de bandas del stack biestacional (24 bandas)."""
    s2 = list(cfg["s2"]["bandas"])
    s2_seca = [f"{b}_SECA" for b in s2]
    s2_lluvia = [f"{b}_LLUVIA" for b in s2]
    sar_seca = ["VV_SECA", "VH_SECA"]
    sar_lluvia = ["VV_LLUVIA", "VH_LLUVIA"]
    todas = s2_seca + s2_lluvia + sar_seca + sar_lluvia
    return {
        "s2_seca": s2_seca,
        "s2_lluvia": s2_lluvia,
        "s2_todas": s2_seca + s2_lluvia,
        "sar_seca": sar_seca,
        "sar_lluvia": sar_lluvia,
        "sar_todas": sar_seca + sar_lluvia,
        "seca_todas": s2_seca + sar_seca,
        "lluvia_todas": s2_lluvia + sar_lluvia,
        "todas": todas,
    }


def init_gee(cfg: Dict[str, Any], force_auth: bool = False) -> None:
    """Inicializa Google Earth Engine usando el proyecto definido en config.yaml."""
    import ee

    project = cfg["gee"]["project"]
    if force_auth:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
        print(f"GEE inicializado correctamente en el proyecto: {project}")
    except Exception as exc:
        print(
            f"No se pudo inicializar GEE directamente ({exc}). "
            "Ejecutando ee.Authenticate()..."
        )
        ee.Authenticate()
        ee.Initialize(project=project)
        print(f"GEE autenticado e inicializado en el proyecto: {project}")


def check_asset_exists(asset_id: str) -> Dict[str, Any]:
    """Verifica si un Asset existe en GEE y retorna sus metadatos básicos."""
    import ee

    try:
        info = ee.data.getAsset(asset_id)
        asset_type = info.get("type", "UNKNOWN")
        summary: Dict[str, Any] = {
            "asset_id": asset_id,
            "existe": True,
            "tipo": asset_type,
        }
        if asset_type == "IMAGE":
            img = ee.Image(asset_id)
            summary["bandas"] = img.bandNames().getInfo()
        elif asset_type == "TABLE":
            fc = ee.FeatureCollection(asset_id)
            summary["n_features"] = fc.size().getInfo()
            first = fc.first()
            summary["propiedades"] = first.propertyNames().getInfo() if summary["n_features"] > 0 else []
        return summary
    except Exception as exc:
        return {
            "asset_id": asset_id,
            "existe": False,
            "tipo": None,
            "error": str(exc),
        }


def load_aoi_geometry(asset_id: str):
    """Carga un AOI desde un Asset de GEE (FeatureCollection o Image) y devuelve ee.Geometry."""
    import ee

    info = ee.data.getAsset(asset_id)
    asset_type = info.get("type", "TABLE")
    if asset_type == "TABLE":
        return ee.FeatureCollection(asset_id).geometry()
    if asset_type == "IMAGE":
        return ee.Image(asset_id).geometry()
    raise ValueError(f"Tipo de asset no soportado para AOI ({asset_id}): {asset_type}")
