"""Pruebas unitarias y end-to-end del pipeline 100% local en SEPAL.io (AOI, S2, S1, Stack, RF y CEO)."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from PIL import Image
from rasterio.transform import Affine
from shapely.geometry import Point, box
from sklearn.ensemble import RandomForestClassifier

from src.config_utils import get_class_maps, get_predictor_band_names, load_config, resolve_max_workers
from src.local_aoi import compute_region_grid, rasterize_aoi_masks
from src.local_rf import (
    build_agreement_stratum_geotiff,
    cross_points_with_local_maps,
    extract_bands_at_points,
    generate_stratified_sample_ceo,
    predict_rf_to_geotiff,
    propose_mangle_candidates_local,
)
from src.local_s1 import apply_local_speckle_filter, refined_lee_7x7_numpy
from src.local_s2 import _resolve_scale_offset, compute_s2_clear_mask
from src.local_stack import add_local_raster_overlay, assemble_local_biseasonal_stack, diagnose_local_stack
from src.metrics import stehman_generalized_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Configuracion de paralelismo y grilla UTM exacta (AOI + doble anillo)
# ---------------------------------------------------------------------------
def test_resolve_max_workers_and_grid_snapping():
    cfg = load_config(REPO_ROOT / "config.yaml")
    w_auto = resolve_max_workers(cfg, cap=8)
    assert 1 <= w_auto <= 8

    cfg_fixed = {"computo": {"max_workers": 3}}
    assert resolve_max_workers(cfg_fixed, cap=16) == 3

    # Poligono sintetico en la costa Pacifica de Guatemala (EPSG:4326)
    poly_4326 = box(-90.525, 13.910, -90.515, 13.920)
    aoi_gdf = gpd.GeoDataFrame({"nombre": ["TEST_AOI"]}, geometry=[poly_4326], crs="EPSG:4326")

    grid = compute_region_grid(aoi_gdf, crs="EPSG:32615", scale_m=10.0, buffer_m=100.0, margen_kernel_m=60.0)
    xmin, ymin, xmax, ymax = grid["bounds_utm"]
    assert xmin % 10.0 == pytest.approx(0.0)
    assert ymin % 10.0 == pytest.approx(0.0)
    assert xmax % 10.0 == pytest.approx(0.0)
    assert ymax % 10.0 == pytest.approx(0.0)
    assert grid["transform"].a == pytest.approx(10.0)
    assert grid["transform"].e == pytest.approx(-10.0)
    assert grid["width"] > 20 and grid["height"] > 20
    assert isinstance(grid["wkt_search_4326"], str) and "POLYGON" in grid["wkt_search_4326"]

    mask_base, mask_buffer = rasterize_aoi_masks(grid)
    assert mask_base.shape == (grid["height"], grid["width"])
    assert mask_base.any()
    assert mask_buffer.sum() > mask_base.sum()
    assert np.all(mask_buffer[mask_base])


# ---------------------------------------------------------------------------
# 2. Sentinel-2 C1 L2A: verificacion anti-doble-offset y mascara MNDWI/SCL
# ---------------------------------------------------------------------------
def test_s2_resolve_scale_offset_prevents_double_subtraction():
    asset_extra = {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
    scl_water = np.full((10, 10), 6, dtype=np.uint8)

    # Caso C1 oficial con +1000 BOA offset: agua tiene DN ~ 1200 -> 1200*1e-4 - 0.1 = +0.02 >= -0.02
    red_c1 = np.full((10, 10), 1200, dtype=np.uint16)
    sc, off = _resolve_scale_offset(asset_extra, red_dn=red_c1, scl=scl_water)
    assert sc == pytest.approx(0.0001)
    assert off == pytest.approx(-0.1)

    # Caso anomalo legacy sin +1000 offset: agua tiene DN ~ 200 -> 200*1e-4 - 0.1 = -0.08 < -0.02
    red_no_offset = np.full((10, 10), 200, dtype=np.uint16)
    sc2, off2 = _resolve_scale_offset(asset_extra, red_dn=red_no_offset, scl=scl_water)
    assert sc2 == pytest.approx(0.0001)
    assert off2 == pytest.approx(0.0)


def test_s2_clear_mask_protects_flooded_mangrove_and_water():
    cfg = load_config(REPO_ROOT / "config.yaml")
    shape = (11, 11)
    valid = np.ones(shape, dtype=bool)
    scl = np.full(shape, 4, dtype=np.uint8)  # vegetacion despejada por defecto
    b3 = np.full(shape, 0.08, dtype=np.float32)
    b11 = np.full(shape, 0.12, dtype=np.float32)
    cloud_prob = np.zeros(shape, dtype=np.float32)

    # Nube gruesa en (5, 5)
    scl[5, 5] = 8
    cloud_prob[5, 5] = 85.0

    # Manglar inundado / canal estuarino en (1, 1): alta MNDWI (>= 0.10) y falsa sombra SCL=3
    scl[1, 1] = 3
    b3[1, 1] = 0.12
    b11[1, 1] = 0.02

    # Agua abierta SCL=6 justo al lado de la zona dilatada de la nube (3, 5)
    scl[3, 5] = 6
    b3[3, 5] = 0.07
    b11[3, 5] = 0.01

    clear = compute_s2_clear_mask(b3, b11, scl, cloud_prob, valid, cfg["s2"])
    assert clear[5, 5] == False  # noqa: E712  (nube bloqueada)
    assert clear[5, 6] == False  # noqa: E712  (buffer 2 px alrededor de nube en tierra bloqueado)
    assert clear[1, 1] == True   # noqa: E712  (manglar inundado protegido contra falsa sombra SCL=3)
    assert clear[3, 5] == True   # noqa: E712  (agua estuarina no contaminada por dilatacion de nube)
    assert clear[9, 9] == True   # noqa: E712  (tierra despejada intacta)


# ---------------------------------------------------------------------------
# 3. Sentinel-1: Refined Lee 7x7 vectorizado, preservacion de bordes y NaNs
# ---------------------------------------------------------------------------
def test_s1_refined_lee_preserves_constant_edges_and_reduces_speckle():
    const = np.full((32, 32), 0.25, dtype=np.float32)
    out_const = refined_lee_7x7_numpy(const)
    assert np.allclose(out_const, 0.25, atol=1e-5)

    # Speckle Gamma multiplicativo (4 looks) alrededor de media 0.25
    rng = np.random.default_rng(2025)
    noisy = rng.gamma(shape=4.0, scale=0.25 / 4.0, size=(64, 64)).astype(np.float32)
    out_lee = refined_lee_7x7_numpy(noisy)
    assert out_lee.dtype == np.float32
    assert float(np.nanstd(out_lee)) < 0.50 * float(np.nanstd(noisy))
    assert float(np.nanmean(out_lee)) == pytest.approx(float(np.nanmean(noisy)), rel=0.05)

    # Borde nitido: Refined Lee debe difuminar menos el borde que un Boxcar 7x7 isotropo
    step = np.full((48, 48), 0.05, dtype=np.float32)
    step[:, 24:] = 0.50
    lee_edge = apply_local_speckle_filter(step, "REFINED_LEE", 7)
    box_edge = apply_local_speckle_filter(step, "BOXCAR", 7)
    err_lee = np.mean(np.abs(lee_edge[10:-10, 21:27] - step[10:-10, 21:27]))
    err_box = np.mean(np.abs(box_edge[10:-10, 21:27] - step[10:-10, 21:27]))
    assert err_lee < err_box

    # Seguridad ante NaNs / ceros: no contamina pixeles lejanos
    with_nan = const.copy()
    with_nan[10:13, 10:13] = np.nan
    with_nan[2, 2] = 0.0
    out_nan = refined_lee_7x7_numpy(with_nan)
    assert np.isnan(out_nan[11, 11])
    assert np.isnan(out_nan[2, 2])
    assert np.isfinite(out_nan[25, 25]) and out_nan[25, 25] == pytest.approx(0.25, abs=1e-4)

    out_none = apply_local_speckle_filter(with_nan, "NONE", 7)
    assert np.isnan(out_none[2, 2]) and out_none[25, 25] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 4. Pipeline completo sintetico en disco: Stack -> Muestreo -> RF -> Estratos -> CEO -> Stehman
# ---------------------------------------------------------------------------
def _write_multiband_tif(path: Path, arr: np.ndarray, crs: str, transform: Affine, band_names: list[str], dtype: str, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype))
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def test_end_to_end_local_pipeline_synthetic(tmp_path: Path):
    cfg = load_config(REPO_ROOT / "config.yaml")
    bands_cfg = get_predictor_band_names(cfg)
    _, _, class_colors = get_class_maps(cfg)

    crs = "EPSG:32615"
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1550000.0)
    h, w = 60, 60

    # Mascara base con margen interior de 2 px respecto al buffer completo
    mask_buffer = np.ones((h, w), dtype=bool)
    mask_base = np.zeros((h, w), dtype=bool)
    mask_base[2:-2, 2:-2] = True

    # 5 franjas verticales bien separadas: clase 1 (cols 0..11), 2 (12..23), 3 (24..35), 4 (36..47), 5 (48..59)
    true_map = np.zeros((h, w), dtype=np.uint8)
    for c_idx in range(5):
        true_map[:, c_idx * 12 : (c_idx + 1) * 12] = c_idx + 1

    # Firmas sinteticas distintivas por clase (10 bandas S2 y 2 bandas SAR por temporada)
    s2_seca = np.zeros((10, h, w), dtype=np.float32)
    s2_lluvia = np.zeros((10, h, w), dtype=np.float32)
    s1_seca = np.zeros((2, h, w), dtype=np.float32)
    s1_lluvia = np.zeros((2, h, w), dtype=np.float32)

    rng = np.random.default_rng(123)
    for cid in range(1, 6):
        m = true_map == cid
        for b in range(10):
            base_s = 0.04 * cid + 0.005 * b
            s2_seca[b, m] = base_s + rng.normal(0.0, 0.002, size=int(m.sum()))
            s2_lluvia[b, m] = base_s + 0.01 + rng.normal(0.0, 0.002, size=int(m.sum()))
        s1_seca[0, m] = -18.0 + 2.5 * cid + rng.normal(0.0, 0.1, size=int(m.sum()))
        s1_seca[1, m] = -24.0 + 2.5 * cid + rng.normal(0.0, 0.1, size=int(m.sum()))
        s1_lluvia[0, m] = -17.5 + 2.5 * cid + rng.normal(0.0, 0.1, size=int(m.sum()))
        s1_lluvia[1, m] = -23.5 + 2.5 * cid + rng.normal(0.0, 0.1, size=int(m.sum()))

    # En la columna 12..14 hacemos que SAR parezca Mangle mientras S2 sigue siendo Bosque -> fuerza desacuerdo (estrato 6)
    s1_seca[0, :, 12:15] = -15.5
    s1_seca[1, :, 12:15] = -21.5
    s1_lluvia[0, :, 12:15] = -15.0
    s1_lluvia[1, :, 12:15] = -21.0

    nobs = np.full((1, h, w), 6, dtype=np.uint16)

    p_s2s = tmp_path / "s2s.tif"
    p_s2l = tmp_path / "s2l.tif"
    p_s1s = tmp_path / "s1s.tif"
    p_s1l = tmp_path / "s1l.tif"
    p_n2s = tmp_path / "n2s.tif"
    p_n2l = tmp_path / "n2l.tif"
    p_n1s = tmp_path / "n1s.tif"
    p_n1l = tmp_path / "n1l.tif"
    p_guia = tmp_path / "guia.tif"

    _write_multiband_tif(p_s2s, s2_seca, crs, transform, bands_cfg["s2_seca"], "float32", np.nan)
    _write_multiband_tif(p_s2l, s2_lluvia, crs, transform, bands_cfg["s2_lluvia"], "float32", np.nan)
    _write_multiband_tif(p_s1s, s1_seca, crs, transform, bands_cfg["sar_seca"], "float32", np.nan)
    _write_multiband_tif(p_s1l, s1_lluvia, crs, transform, bands_cfg["sar_lluvia"], "float32", np.nan)
    _write_multiband_tif(p_n2s, nobs, crs, transform, ["NOBS_S2_SECA"], "uint16", 0)
    _write_multiband_tif(p_n2l, nobs, crs, transform, ["NOBS_S2_LLUVIA"], "uint16", 0)
    _write_multiband_tif(p_n1s, nobs, crs, transform, ["NOBS_S1_SECA"], "uint16", 0)
    _write_multiband_tif(p_n1l, nobs, crs, transform, ["NOBS_S1_LLUVIA"], "uint16", 0)
    guia_arr = np.where(true_map == 1, 1, 0)[None, :, :]
    _write_multiband_tif(p_guia, guia_arr, crs, transform, ["b1"], "uint8", 0)

    stack_path = tmp_path / "stack.tif"
    qa_path = tmp_path / "qa.tif"
    assemble_local_biseasonal_stack(
        p_s2s, p_n2s, p_s2l, p_n2l, p_s1s, p_n1s, p_s1l, p_n1l,
        mask_base=mask_base,
        mask_buffer=mask_buffer,
        band_names_24=bands_cfg["todas"],
        out_stack_path=stack_path,
        out_qa_path=qa_path,
        block_rows=16,
    )

    diag = diagnose_local_stack(stack_path, qa_path)
    assert diag["n_bandas_stack"] == 24
    assert diag["fraccion_valida_aoi"] == pytest.approx(1.0)
    assert diag["tamano_stack_mb"] > 0

    # Candidatos Mangle desde DinMangle local erosionado (y caso borde n_points=1)
    cand = propose_mangle_candidates_local(p_guia, qa_path, "PACIFICO", n_points=10, erosion_m=10.0)
    assert len(cand) == 10 and (cand["clase_id"] == 1).all()
    cand_one = propose_mangle_candidates_local(p_guia, qa_path, "PACIFICO", n_points=1, erosion_m=10.0)
    assert len(cand_one) == 1

    # Construir puntos de entrenamiento en EPSG:4326 en los centros de cada franja limpia
    recs = []
    for cid in range(1, 6):
        c_center = cid * 12 - 6
        for r_idx in range(6, 54, 4):
            x, y = rasterio.transform.xy(transform, r_idx, c_center, offset="center")
            recs.append({"clase_id": cid, "geometry": Point(x, y)})
    pts_utm = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs).to_crs("EPSG:4326")

    pts_single = extract_bands_at_points(stack_path, pts_utm.iloc[[0]], qa_path=qa_path, require_base_aoi=True)
    assert len(pts_single) == 1 and all(b in pts_single.columns for b in bands_cfg["todas"])

    pts_feat = extract_bands_at_points(stack_path, pts_utm, qa_path=qa_path, require_base_aoi=True)
    assert len(pts_feat) == len(pts_utm)

    ms_feats = bands_cfg["s2_seca"] + bands_cfg["s2_lluvia"]
    sar_feats = bands_cfg["sar_seca"] + bands_cfg["sar_lluvia"]
    comb_feats = bands_cfg["todas"]

    rf_ms = RandomForestClassifier(n_estimators=20, random_state=1).fit(pts_feat[ms_feats].values, pts_feat["clase_id"].values)
    rf_sar = RandomForestClassifier(n_estimators=20, random_state=1).fit(pts_feat[sar_feats].values, pts_feat["clase_id"].values)
    rf_comb = RandomForestClassifier(n_estimators=20, random_state=1).fit(pts_feat[comb_feats].values, pts_feat["clase_id"].values)

    m_ms = tmp_path / "map_ms.tif"
    m_sar = tmp_path / "map_sar.tif"
    m_comb = tmp_path / "map_comb.tif"
    area_ms = predict_rf_to_geotiff(rf_ms, stack_path, qa_path, ms_feats, m_ms, class_colors, block_size=20)
    predict_rf_to_geotiff(rf_sar, stack_path, qa_path, sar_feats, m_sar, class_colors, block_size=20)
    area_comb = predict_rf_to_geotiff(rf_comb, stack_path, qa_path, comb_feats, m_comb, class_colors, block_size=20)

    expected_base_ha = float(mask_base.sum() * 100.0 / 10000.0)
    assert sum(area_ms.values()) == pytest.approx(expected_base_ha)
    assert sum(area_comb.values()) == pytest.approx(expected_base_ha)

    # Mapa de estratos y generacion de muestra CEO
    m_est = tmp_path / "estratos.tif"
    areas_est = build_agreement_stratum_geotiff(m_ms, m_sar, m_comb, m_est, block_size=20)
    assert 6 in areas_est and areas_est[6] > 0.0
    assert sum(areas_est.values()) == pytest.approx(expected_base_ha)

    ceo_csv = tmp_path / "diseno_ceo.csv"
    ceo_gpkg = tmp_path / "diseno_ceo.gpkg"
    gdf_pts, plan_df = generate_stratified_sample_ceo(
        m_est, ceo_csv, out_gpkg_path=ceo_gpkg, stratum_areas_ha=areas_est, min_per_stratum=8, total_n=60, seed=42
    )
    assert ceo_csv.is_file() and ceo_gpkg.is_file()
    assert list(pd.read_csv(ceo_csv).columns) == ["PLOTID", "LON", "LAT", "stratum"]
    assert plan_df["W_i"].sum() == pytest.approx(1.0)
    assert len(gdf_pts) == 60

    # Simular fotointerpretacion CEO con verdad de campo y ejecutar Stehman (2014)
    pts_utm_ceo = gdf_pts.to_crs(crs)
    gt_vals = []
    for geom in pts_utm_ceo.geometry:
        r_idx, c_idx = rasterio.transform.rowcol(transform, geom.x, geom.y)
        gt_vals.append(int(true_map[r_idx, c_idx]))
    gdf_pts["ref"] = gt_vals

    crossed = cross_points_with_local_maps(
        gdf_pts[["PLOTID", "LON", "LAT", "stratum", "ref"]],
        lon_col="LON",
        lat_col="LAT",
        map_paths={"MS": m_ms, "SAR": m_sar, "MS_SAR_FULL": m_comb},
    )
    assert len(crossed) == len(gdf_pts)

    res_comb = stehman_generalized_accuracy(
        df=crossed,
        stratum_col="stratum",
        pred_col="pred_MS_SAR_FULL",
        ref_col="ref",
        stratum_areas_ha=areas_est,
        class_labels=[1, 2, 3, 4, 5],
        class_names={1: "Mangle", 2: "Bosque latifoliado", 3: "Agua", 4: "Infraestructura urbana", 5: "Pastos/Ganadería"},
        n_bootstrap=100,
    )
    assert res_comb["OA"] >= 0.85
    assert res_comb["per_class"]["F1"].notna().all()
    assert res_comb["per_class"]["area_ajustada_ha"].sum() == pytest.approx(expected_base_ha, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Smoke test de visualizacion EPSG:3857 base64 PNG (sin localtileserver)
# ---------------------------------------------------------------------------
def test_add_local_raster_overlay_categorical_and_rgb(tmp_path: Path):
    class _DummyMap:
        def __init__(self) -> None:
            self.layers: list = []

        def add_layer(self, layer) -> None:
            self.layers.append(layer)

    crs = "EPSG:32615"
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1550000.0)
    _, _, class_colors = get_class_maps(load_config(REPO_ROOT / "config.yaml"))

    cat_tif = tmp_path / "cat.tif"
    cat_arr = np.tile(np.array([[1, 2, 3, 4, 5]], dtype=np.uint8), (20, 4))[None, :, :]
    _write_multiband_tif(cat_tif, cat_arr, crs, transform, ["clasificacion"], "uint8", 0)

    rgb_tif = tmp_path / "rgb.tif"
    rgb_arr = np.full((3, 20, 20), 0.18, dtype=np.float32)
    _write_multiband_tif(rgb_tif, rgb_arr, crs, transform, ["B8", "B11", "B4"], "float32", np.nan)

    m = _DummyMap()
    ov1 = add_local_raster_overlay(m, cat_tif, bands=(1,), layer_name="Cat", palette=class_colors, categorical=True, visible=False)
    ov2 = add_local_raster_overlay(m, rgb_tif, bands=("B8", "B11", "B4"), layer_name="RGB", stretch=(0.0, 0.35), visible=True)

    assert len(m.layers) == 2
    assert ov1.visible is False and ov2.visible is True
    for ov in (ov1, ov2):
        assert ov.url.startswith("data:image/png;base64,")
        raw = base64.b64decode(ov.url.split(",", 1)[1])
        img = Image.open(BytesIO(raw))
        assert img.mode == "RGBA" and img.size[0] > 0 and img.size[1] > 0
