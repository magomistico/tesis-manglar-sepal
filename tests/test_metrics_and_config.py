"""Pruebas unitarias para configuración, métricas Olofsson/Stehman, F1, hipótesis, gráficos y notebooks."""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.config_utils import get_class_maps, get_predictor_band_names, load_config
from src.metrics import (
    compute_preliminary_metrics,
    kappa_z_test,
    mcnemar_test,
    olofsson_accuracy_area,
    stehman_generalized_accuracy,
    stratified_paired_z_test,
)
from src.plotting import (
    plot_band_contributions,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_model_comparison,
    plot_scree_and_variance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Configuracion y utilidades
# ---------------------------------------------------------------------------
def test_config_loads_and_matches_protocol():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg["computo"]["modo"] == "sepal_local"
    assert cfg["computo"]["block_size"] == 1024
    assert cfg["local_assets"]["aoi_dir"] == "outputs/aoi"
    assert cfg["local_assets"]["guia_dir"] == "outputs/guia"
    assert cfg["local_assets"]["stack_dir"] == "outputs/stack"
    assert cfg["s2"]["stac_collection"] == "sentinel-2-c1-l2a"
    assert cfg["s1"]["fuente_local"] == "OPERA_RTC_S1"
    assert cfg["s1"]["speckle_filter"] == "REFINED_LEE"
    assert cfg["s1"]["kernel_size"] == 7
    assert cfg["validacion_olofsson"]["n_min_por_estrato"] == 50

    assert cfg["gee"]["project"] == "tesismangle"
    assert cfg["gee"]["asset_root"] == "projects/tesismangle/assets/"
    assert cfg["assets"]["mangle_guia"] == "projects/tesismangle/assets/DinMangle_2020_2022_V23"
    assert cfg["assets"]["aoi"]["PACIFICO"] == "projects/tesismangle/assets/AOI_PACIFICO"
    assert cfg["assets"]["aoi"]["CARIBE"] == "projects/tesismangle/assets/AOI_CARIBE"
    assert cfg["regiones"]["PACIFICO"]["crs"] == "EPSG:32615"
    assert cfg["regiones"]["CARIBE"]["crs"] == "EPSG:32616"
    assert cfg["regiones"]["PACIFICO"]["linea_base_marn_2013_ha"] == pytest.approx(17670.56)
    assert cfg["regiones"]["CARIBE"]["linea_base_marn_2013_ha"] == pytest.approx(1169.52)

    nombres, cortos, colores = get_class_maps(cfg)
    assert list(nombres.keys()) == [1, 2, 3, 4, 5]
    assert nombres[1] == "Mangle"
    assert len(cortos) == 5 and len(colores) == 5

    bands = get_predictor_band_names(cfg)
    assert len(bands["s2_seca"]) == 10
    assert len(bands["s2_lluvia"]) == 10
    assert len(bands["sar_seca"]) == 2
    assert len(bands["sar_lluvia"]) == 2
    assert len(bands["todas"]) == 24
    assert bands["todas"][:3] == ["B2_SECA", "B3_SECA", "B4_SECA"]
    assert bands["todas"][-2:] == ["VV_LLUVIA", "VH_LLUVIA"]


# ---------------------------------------------------------------------------
# 2. Metricas preliminares (70/30) y formula PP = VP / (VP + FN)
# ---------------------------------------------------------------------------
def test_preliminary_metrics_hand_calculated():
    # 2 clases: pred = [1, 1, 1, 2, 2], true = [1, 1, 2, 1, 2]
    # Para clase 1: VP=2, FP=1, FN=1 -> PU = 2/3, PP = 2/3, F1 = 2/3
    # Para clase 2: VP=1, FP=1, FN=1 -> PU = 1/2, PP = 1/2, F1 = 1/2
    # OA = 3/5 = 0.6; Pe = (3*3 + 2*2)/25 = 13/25 = 0.52; Kappa = (0.6 - 0.52)/(1 - 0.52) = 1/6
    y_true = [1, 1, 2, 1, 2]
    y_pred = [1, 1, 1, 2, 2]
    res = compute_preliminary_metrics(y_true, y_pred, [1, 2], {1: "Mangle", 2: "Latifoliado"})

    assert res["n_muestras"] == 5
    assert res["OA"] == pytest.approx(0.6)
    assert res["Pe"] == pytest.approx(0.52)
    assert res["Kappa"] == pytest.approx(1.0 / 6.0)

    c1 = res["per_class"].set_index("clase_id").loc[1]
    assert c1["VP"] == 2 and c1["FP"] == 1 and c1["FN"] == 1
    assert c1["PU"] == pytest.approx(2.0 / 3.0)
    # Verifica expresamente PP = VP / (VP + FN) y NO la errata VP / (VP - FN)
    assert c1["PP"] == pytest.approx(2.0 / 3.0)
    assert c1["F1"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. Olofsson et al. (2014) - Ejemplo publicado (Tabla 8, RSE 148:42-57)
# ---------------------------------------------------------------------------
def test_olofsson_2014_published_example():
    # Tabla 8 de Olofsson et al. (2014), p. 54:
    # Clases: 1=Deforestacion, 2=Ganancia, 3=Bosque estable, 4=No-bosque estable
    counts = np.array(
        [
            [66, 0, 5, 4],
            [0, 55, 8, 12],
            [1, 0, 153, 11],
            [2, 1, 9, 313],
        ],
        dtype=float,
    )
    areas_ha = [200_000.0, 150_000.0, 3_200_000.0, 6_450_000.0]
    res = olofsson_accuracy_area(
        conf_counts=counts,
        stratum_areas_ha=areas_ha,
        class_labels=[1, 2, 3, 4],
        n_bootstrap=300,
        seed=42,
    )

    # Valores publicados en Olofsson et al. (2014) Seccion 5.1 / Tabla 8:
    # W = [0.02, 0.015, 0.32, 0.645]
    # p11 = 0.02 * 66/75 = 0.0176; p22 = 0.015 * 55/75 = 0.0110
    # p33 = 0.32 * 153/165 = 0.29672727; p44 = 0.645 * 313/325 = 0.62118462
    expected_oa = 0.02 * (66 / 75) + 0.015 * (55 / 75) + 0.32 * (153 / 165) + 0.645 * (313 / 325)
    assert res["OA"] == pytest.approx(expected_oa, abs=1e-10)
    assert res["OA"] == pytest.approx(0.9465, abs=1e-4)

    pc = res["per_class"].set_index("clase_id")
    # Clase 1 (Deforestacion): PU = 66/75 = 0.88; area ajustada ~ 235,086 ha (p_.1 ~ 0.023509)
    assert pc.loc[1, "PU"] == pytest.approx(66.0 / 75.0, abs=1e-10)
    expected_p_col1 = (
        0.02 * (66 / 75) + 0.015 * 0.0 + 0.32 * (1 / 165) + 0.645 * (2 / 325)
    )
    assert pc.loc[1, "area_ajustada_ha"] == pytest.approx(10_000_000.0 * expected_p_col1, rel=1e-10)
    assert pc.loc[1, "PP"] == pytest.approx((0.02 * 66 / 75) / expected_p_col1, abs=1e-10)
    # F1 exacto = 2*PU*PP/(PU+PP) y dentro de su IC bootstrap
    expected_f1_c1 = 2 * pc.loc[1, "PU"] * pc.loc[1, "PP"] / (pc.loc[1, "PU"] + pc.loc[1, "PP"])
    assert pc.loc[1, "F1"] == pytest.approx(expected_f1_c1, abs=1e-10)
    assert pc.loc[1, "F1_CI_low"] <= pc.loc[1, "F1"] <= pc.loc[1, "F1_CI_high"]


# ---------------------------------------------------------------------------
# 4. Equivalencia exacta Stehman (2014) == Olofsson (2014) cuando estrato == mapa
# ---------------------------------------------------------------------------
def test_stehman_reduces_exactly_to_olofsson():
    counts = np.array(
        [
            [45, 3, 1, 0, 1],
            [4, 40, 1, 0, 5],
            [1, 0, 48, 0, 1],
            [0, 1, 0, 46, 3],
            [2, 4, 0, 2, 42],
        ],
        dtype=int,
    )
    labels = [1, 2, 3, 4, 5]
    areas = [18000.0, 25000.0, 15000.0, 5000.0, 37000.0]

    # Expandir conteos a tabla de puntos individuales (formato CEO)
    rows = []
    for i, ci in enumerate(labels):
        for j, cj in enumerate(labels):
            for _ in range(counts[i, j]):
                rows.append({"stratum": ci, "pred": ci, "ref": cj})
    df_pts = pd.DataFrame(rows)

    res_olof = olofsson_accuracy_area(counts, areas, labels, n_bootstrap=0)
    res_steh = stehman_generalized_accuracy(
        df=df_pts,
        stratum_col="stratum",
        pred_col="pred",
        ref_col="ref",
        stratum_areas_ha=dict(zip(labels, areas)),
        class_labels=labels,
        n_bootstrap=0,
    )

    assert res_steh["OA"] == pytest.approx(res_olof["OA"], abs=1e-12)
    assert res_steh["OA_SE"] == pytest.approx(res_olof["OA_SE"], abs=1e-12)
    assert res_steh["Kappa"] == pytest.approx(res_olof["Kappa"], abs=1e-12)

    res_steh_same_col = stehman_generalized_accuracy(
        df=df_pts,
        stratum_col="stratum",
        pred_col="stratum",
        ref_col="ref",
        stratum_areas_ha=dict(zip(labels, areas)),
        class_labels=labels,
        n_bootstrap=0,
    )
    assert res_steh_same_col["OA"] == pytest.approx(res_olof["OA"], abs=1e-12)

    df_o = res_olof["per_class"].set_index("clase_id")
    df_s = res_steh["per_class"].set_index("clase_id")
    for col in ["PU", "PU_SE", "PP", "PP_SE", "F1", "area_ajustada_ha", "area_SE_ha"]:
        np.testing.assert_allclose(df_s[col].values, df_o[col].values, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Stehman (2014) con 6 estratos de Acuerdo/Desacuerdo + Pruebas de Hipotesis
# ---------------------------------------------------------------------------
def test_stehman_six_strata_and_hypothesis_tests():
    # Simula 6 estratos (1..5 acuerdo + 6 desacuerdo de mangle) y 2 modelos (A superior a B)
    rng = np.random.default_rng(12345)
    stratum_areas = {1: 15000.0, 2: 20000.0, 3: 10000.0, 4: 4000.0, 5: 25000.0, 6: 6000.0}
    records = []
    # Estratos 1..5: ambos modelos suelen coincidir con la referencia
    for h in [1, 2, 3, 4, 5]:
        for k in range(40):
            ref = h if k < 36 else (2 if h == 1 else 1)
            pred_a = h
            pred_b = h if k < 35 else (2 if h == 1 else 5)
            records.append({"stratum": h, "ref": ref, "pred_A": pred_a, "pred_B": pred_b})
    # Estrato 6 (zona de desacuerdo): Modelo A (MS+SAR) acierta mucho mas que Modelo B (solo MS)
    for k in range(60):
        ref = 1 if k < 48 else 2
        pred_a = 1 if k < 46 else 2
        pred_b = 2 if k < 45 else 1
        records.append({"stratum": 6, "ref": ref, "pred_A": pred_a, "pred_B": pred_b})

    df = pd.DataFrame(records)
    res_a = stehman_generalized_accuracy(
        df, "stratum", "pred_A", "ref", stratum_areas, [1, 2, 3, 4, 5], n_bootstrap=300, seed=7
    )
    res_b = stehman_generalized_accuracy(
        df, "stratum", "pred_B", "ref", stratum_areas, [1, 2, 3, 4, 5], n_bootstrap=300, seed=7
    )

    assert res_a["OA"] > res_b["OA"]
    assert res_a["per_class"].loc[0, "F1"] > res_b["per_class"].loc[0, "F1"]
    assert res_a["per_class"]["area_ajustada_ha"].sum() == pytest.approx(sum(stratum_areas.values()), rel=1e-9)

    mc_mgl = mcnemar_test(df["ref"], df["pred_A"], df["pred_B"], binary_target_class=1)
    assert mc_mgl["rechaza_H0"] is True
    assert mc_mgl["p_value_recomendado"] < 0.001

    z_pair = stratified_paired_z_test(df, "stratum", "pred_A", "pred_B", "ref", stratum_areas)
    assert z_pair["delta_OA_A_menos_B"] > 0
    assert z_pair["rechaza_H0"] is True

    kz = kappa_z_test(res_a["Kappa"], res_a["Kappa_SE"], res_b["Kappa"], res_b["Kappa_SE"])
    assert kz["delta_kappa"] > 0

    # Control negativo: comparar un modelo consigo mismo NO debe rechazar H0
    mc_same = mcnemar_test(df["ref"], df["pred_A"], df["pred_A"])
    assert mc_same["rechaza_H0"] is False
    assert mc_same["p_value_recomendado"] == pytest.approx(1.0)
    z_same = stratified_paired_z_test(df, "stratum", "pred_A", "pred_A", "ref", stratum_areas)
    assert z_same["rechaza_H0"] is False


# ---------------------------------------------------------------------------
# 6. Prueba de renderizado de figuras 300 DPI (plotting.py)
# ---------------------------------------------------------------------------
def test_plotting_smoke(tmp_path: Path):
    corr = pd.DataFrame(np.eye(4), index=["B2_SECA", "B8_SECA", "VV_SECA", "VH_SECA"], columns=["B2_SECA", "B8_SECA", "VV_SECA", "VH_SECA"])
    p1 = tmp_path / "corr.png"
    plot_correlation_matrix(corr, "Test Corr", save_path=p1)
    assert p1.stat().st_size > 0

    p2 = tmp_path / "scree.png"
    plot_scree_and_variance([2.5, 1.1, 0.3, 0.1], [0.625, 0.275, 0.075, 0.025], "Test Scree", save_path=p2)
    assert p2.stat().st_size > 0

    contrib = pd.DataFrame({"contribucion_pca_pct": [30.0, 25.0, 25.0, 20.0]}, index=["B8_SECA", "B11_SECA", "VH_LLUVIA", "VV_SECA"])
    p3 = tmp_path / "contrib.png"
    plot_band_contributions(contrib, "contribucion_pca_pct", "Test Contrib", save_path=p3)
    assert p3.stat().st_size > 0

    cm = pd.DataFrame([[40, 2], [3, 45]], index=["Mangle", "Agua"], columns=["Mangle", "Agua"])
    p4 = tmp_path / "cm.png"
    plot_confusion_matrix(cm, "Test CM", is_proportion=False, save_path=p4)
    assert p4.stat().st_size > 0

    summary = pd.DataFrame(
        {
            "Modelo": ["MS", "SAR", "MS_SAR_PCA"],
            "OA": [0.85, 0.72, 0.91],
            "OA_CI_half": [0.03, 0.04, 0.02],
            "Kappa": [0.80, 0.65, 0.88],
            "Kappa_CI_half": [0.04, 0.05, 0.03],
            "F1_Mangle": [0.84, 0.70, 0.92],
            "F1_Mangle_CI_half": [0.03, 0.05, 0.02],
        }
    )
    p5 = tmp_path / "comp.png"
    plot_model_comparison(summary, "Test Comp", save_path=p5)
    assert p5.stat().st_size > 0


# ---------------------------------------------------------------------------
# 7. Integridad sintactica de todos los modulos src/*.py y notebooks/*.ipynb
# ---------------------------------------------------------------------------
def test_all_python_modules_and_notebooks_valid_syntax():
    py_files = sorted((REPO_ROOT / "src").glob("*.py"))
    assert len(py_files) == 12
    for pyf in py_files:
        py_compile.compile(str(pyf), doraise=True)

    nb_files = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert len(nb_files) == 6
    for nbf in nb_files:
        nb = json.loads(nbf.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 5
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    pytest.fail(f"Error de sintaxis en {nbf.name} celda {idx}: {exc}")
