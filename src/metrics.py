"""Métricas de precisión, validación Olofsson (2014) / Stehman (2014), F1-Score y pruebas de hipótesis.

Convención de matriz de confusión (alineada al Protocolo FINAL, pp. 48-49):
- Filas (i): clase predicha por el mapa / clasificación.
- Columnas (j): clase de referencia (terreno verdadero / Collect Earth Online).
- Precisión de Usuario (PU) = VP / (VP + FP)
- Precisión de Productor (PP) = VP / (VP + FN)   [fórmula estándar corregida]
- F1-Score = 2 * PU * PP / (PU + PP)

Referencias:
- Olofsson, P., et al. (2014). Good practices for estimating area and assessing accuracy
  of land change. Remote Sensing of Environment, 148, 42-57.
- Stehman, S. V. (2014). Estimating area and map accuracy for stratified random sampling
  when the strata are different from the map classes. Int. J. Remote Sensing, 35(13), 4923-4939.
- Congalton, R. G., & Green, K. (2019). Assessing the Accuracy of Remotely Sensed Data.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _z_value(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


def _safe_f1(pu: float, pp: float) -> float:
    if np.isnan(pu) or np.isnan(pp) or (pu + pp) == 0:
        return np.nan
    return float(2.0 * pu * pp / (pu + pp))


def _kappa_from_proportions(p_mat: np.ndarray) -> float:
    po = float(np.trace(p_mat))
    pe = float(np.sum(p_mat.sum(axis=1) * p_mat.sum(axis=0)))
    if np.isclose(1.0 - pe, 0.0):
        return np.nan
    return float((po - pe) / (1.0 - pe))


def compute_preliminary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_labels: Sequence[int],
    class_names: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    """Calcula métricas preliminares no ponderadas (diagnóstico 70/30).

    Filas = predicho, Columnas = referencia.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    labels = [int(c) for c in class_labels]
    k = len(labels)
    idx = {c: i for i, c in enumerate(labels)}

    cm = np.zeros((k, k), dtype=int)
    for yt, yp in zip(y_true_arr, y_pred_arr):
        if yp in idx and yt in idx:
            cm[idx[yp], idx[yt]] += 1

    n_total = int(cm.sum())
    if n_total == 0:
        raise ValueError("La matriz de confusión está vacía.")

    po = float(np.trace(cm) / n_total)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n_total ** 2))
    kappa = float((po - pe) / (1.0 - pe)) if not np.isclose(1.0 - pe, 0.0) else np.nan

    # Varianza asintótica de Kappa (Congalton & Green, 2019)
    p_mat = cm / n_total
    p_row = row_sums / n_total
    p_col = col_sums / n_total
    theta1 = po
    theta2 = pe
    theta3 = float(np.sum(np.diag(p_mat) * (p_row + p_col)))
    theta4 = float(np.sum(p_mat * ((p_row[:, None] + p_col[None, :]) ** 2)))
    if not np.isclose(1.0 - theta2, 0.0):
        t1 = theta1 * (1.0 - theta1) / ((1.0 - theta2) ** 2)
        t2 = 2.0 * (1.0 - theta1) * (2.0 * theta1 * theta2 - theta3) / ((1.0 - theta2) ** 3)
        t3 = ((1.0 - theta1) ** 2) * (theta4 - 4.0 * (theta2 ** 2)) / ((1.0 - theta2) ** 4)
        kappa_var = max(0.0, float((t1 + t2 + t3) / n_total))
    else:
        kappa_var = np.nan

    rows = []
    for i, c in enumerate(labels):
        vp = int(cm[i, i])
        fp = int(row_sums[i] - vp)
        fn = int(col_sums[i] - vp)
        pu = vp / (vp + fp) if (vp + fp) > 0 else np.nan
        pp = vp / (vp + fn) if (vp + fn) > 0 else np.nan
        f1 = _safe_f1(pu, pp)
        rows.append(
            {
                "clase_id": c,
                "clase": class_names[c] if class_names and c in class_names else str(c),
                "n_predicho": int(row_sums[i]),
                "n_referencia": int(col_sums[i]),
                "VP": vp,
                "FP": fp,
                "FN": fn,
                "PU": pu,
                "PP": pp,
                "F1": f1,
                "error_comision": 1.0 - pu if not np.isnan(pu) else np.nan,
                "error_omision": 1.0 - pp if not np.isnan(pp) else np.nan,
            }
        )

    per_class_df = pd.DataFrame(rows)
    names_axis = [class_names[c] if class_names and c in class_names else str(c) for c in labels]
    cm_df = pd.DataFrame(cm, index=names_axis, columns=names_axis)
    cm_df.index.name = "Predicho"
    cm_df.columns.name = "Referencia"

    macro_f1 = float(per_class_df["F1"].mean(skipna=True))
    weights_ref = per_class_df["n_referencia"].values / n_total
    weighted_f1 = float(np.nansum(per_class_df["F1"].values * weights_ref))

    return {
        "n_muestras": n_total,
        "OA": po,
        "Pe": pe,
        "Kappa": kappa,
        "Kappa_SE": float(np.sqrt(kappa_var)) if not np.isnan(kappa_var) else np.nan,
        "Kappa_var": kappa_var,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "per_class": per_class_df,
        "confusion_matrix": cm_df,
    }


def olofsson_accuracy_area(
    conf_counts: np.ndarray,
    stratum_areas_ha: Sequence[float],
    class_labels: Sequence[int],
    class_names: Dict[int, str] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 20250822,
) -> Dict[str, Any]:
    """Estimadores insesgados de precisión y área según Olofsson et al. (2014).

    Se aplica cuando los estratos de muestreo coinciden con las clases del mapa
    evaluado (filas = estrato/clase del mapa, columnas = clase de referencia CEO).
    Calcula también F1-Score ponderado por área y Kappa con IC 95% vía bootstrap estratificado.
    """
    n_mat = np.asarray(conf_counts, dtype=float)
    areas = np.asarray(stratum_areas_ha, dtype=float)
    labels = [int(c) for c in class_labels]
    k = len(labels)

    if n_mat.shape != (k, k):
        raise ValueError(f"conf_counts debe ser ({k}, {k}), recibido {n_mat.shape}.")
    if areas.shape != (k,):
        raise ValueError(f"stratum_areas_ha debe tener longitud {k}, recibido {areas.shape}.")

    a_tot = float(areas.sum())
    w = areas / a_tot
    n_i = n_mat.sum(axis=1)
    if np.any(n_i < 2):
        raise ValueError("Cada estrato debe tener al menos n_i >= 2 muestras para estimar varianza.")

    z = _z_value(confidence)
    # Matriz de proporciones de área p_ij = W_i * (n_ij / n_i)
    r_mat = n_mat / n_i[:, None]
    p_mat = w[:, None] * r_mat

    p_col = p_mat.sum(axis=0)  # proporción insesgada de área real por clase
    oa = float(np.trace(p_mat))
    u = np.diag(r_mat)  # Precisión de Usuario
    p = np.where(p_col > 0, np.diag(p_mat) / p_col, np.nan)  # Precisión de Productor
    f1 = np.array([_safe_f1(u[j], p[j]) for j in range(k)], dtype=float)
    kappa = _kappa_from_proportions(p_mat)

    # Varianzas exactas (Olofsson et al., 2014, Eqs. 5, 6, 7 y 10)
    var_u = u * (1.0 - u) / (n_i - 1.0)
    se_u = np.sqrt(np.maximum(var_u, 0.0))

    var_oa = float(np.sum((w ** 2) * var_u))
    se_oa = float(np.sqrt(max(var_oa, 0.0)))

    # Varianza de proporción de área p_.j (Eq. 10)
    var_p_col = np.sum((w[:, None] ** 2) * r_mat * (1.0 - r_mat) / (n_i[:, None] - 1.0), axis=0)
    se_p_col = np.sqrt(np.maximum(var_p_col, 0.0))
    area_adj_ha = a_tot * p_col
    se_area_ha = a_tot * se_p_col

    # Varianza de Precisión de Productor (Eq. 7)
    var_p = np.full(k, np.nan, dtype=float)
    for j in range(k):
        if p_col[j] > 0:
            term_diag = (w[j] ** 2) * ((1.0 - p[j]) ** 2) * var_u[j]
            term_off = 0.0
            for i in range(k):
                if i != j:
                    term_off += (w[i] ** 2) * r_mat[i, j] * (1.0 - r_mat[i, j]) / (n_i[i] - 1.0)
            var_p[j] = (term_diag + (p[j] ** 2) * term_off) / (p_col[j] ** 2)
    se_p = np.sqrt(np.maximum(var_p, 0.0))

    # Bootstrap estratificado para IC 95% de F1-Score y Kappa
    rng = np.random.default_rng(seed)
    boot_f1 = np.full((n_bootstrap, k), np.nan, dtype=float)
    boot_kappa = np.full(n_bootstrap, np.nan, dtype=float)
    if n_bootstrap > 0:
        n_i_int = n_i.astype(int)
        for b in range(n_bootstrap):
            p_b = np.zeros((k, k), dtype=float)
            for i in range(k):
                counts_b = rng.multinomial(n_i_int[i], r_mat[i])
                p_b[i] = w[i] * (counts_b / n_i_int[i])
            p_col_b = p_b.sum(axis=0)
            u_b = np.diag(p_b) / w
            pr_b = np.where(p_col_b > 0, np.diag(p_b) / p_col_b, np.nan)
            for j in range(k):
                boot_f1[b, j] = _safe_f1(u_b[j], pr_b[j])
            boot_kappa[b] = _kappa_from_proportions(p_b)

    alpha = 1.0 - confidence
    f1_lo = np.nanpercentile(boot_f1, 100 * alpha / 2.0, axis=0) if n_bootstrap > 0 else np.full(k, np.nan)
    f1_hi = np.nanpercentile(boot_f1, 100 * (1.0 - alpha / 2.0), axis=0) if n_bootstrap > 0 else np.full(k, np.nan)
    f1_se = np.nanstd(boot_f1, axis=0, ddof=1) if n_bootstrap > 1 else np.full(k, np.nan)
    kappa_se = float(np.nanstd(boot_kappa, ddof=1)) if n_bootstrap > 1 else np.nan
    kappa_lo = float(np.nanpercentile(boot_kappa, 100 * alpha / 2.0)) if n_bootstrap > 0 else np.nan
    kappa_hi = float(np.nanpercentile(boot_kappa, 100 * (1.0 - alpha / 2.0))) if n_bootstrap > 0 else np.nan

    rows = []
    names_axis = [class_names[c] if class_names and c in class_names else str(c) for c in labels]
    for j, c in enumerate(labels):
        rows.append(
            {
                "clase_id": c,
                "clase": names_axis[j],
                "area_mapeada_ha": areas[j],
                "W_i": w[j],
                "n_estrato": int(n_i[j]),
                "area_ajustada_ha": area_adj_ha[j],
                "area_SE_ha": se_area_ha[j],
                "area_CI_half_ha": z * se_area_ha[j],
                "area_CI_low_ha": max(0.0, area_adj_ha[j] - z * se_area_ha[j]),
                "area_CI_high_ha": area_adj_ha[j] + z * se_area_ha[j],
                "PU": u[j],
                "PU_SE": se_u[j],
                "PU_CI_half": z * se_u[j],
                "PP": p[j],
                "PP_SE": se_p[j],
                "PP_CI_half": z * se_p[j],
                "F1": f1[j],
                "F1_SE": f1_se[j],
                "F1_CI_low": f1_lo[j],
                "F1_CI_high": f1_hi[j],
            }
        )

    p_df = pd.DataFrame(p_mat, index=names_axis, columns=names_axis)
    p_df.index.name = "Mapa (Estrato)"
    p_df.columns.name = "Referencia (CEO)"

    c_df = pd.DataFrame(n_mat.astype(int), index=names_axis, columns=names_axis)
    c_df.index.name = "Mapa (Estrato)"
    c_df.columns.name = "Referencia (CEO)"

    return {
        "metodo": "Olofsson et al. (2014)",
        "n_muestras": int(n_i.sum()),
        "area_total_ha": a_tot,
        "OA": oa,
        "OA_SE": se_oa,
        "OA_CI_half": z * se_oa,
        "OA_CI_low": max(0.0, oa - z * se_oa),
        "OA_CI_high": min(1.0, oa + z * se_oa),
        "Kappa": kappa,
        "Kappa_SE": kappa_se,
        "Kappa_CI_low": kappa_lo,
        "Kappa_CI_high": kappa_hi,
        "per_class": pd.DataFrame(rows),
        "matrix_proportions": p_df,
        "matrix_counts": c_df,
    }


def _stratified_ratio_estimator(
    y: np.ndarray,
    x: np.ndarray,
    strata: np.ndarray,
    stratum_ids: Sequence[Any],
    stratum_weights: Dict[Any, float],
) -> tuple[float, float]:
    """Estimador de razón estratificado (Stehman, 2014, Eq. 27 / Cochran, 1977)."""
    y_hat = 0.0
    x_hat = 0.0
    stats_h = []
    for h in stratum_ids:
        mask = strata == h
        n_h = int(mask.sum())
        if n_h < 2:
            raise ValueError(f"Estrato '{h}' tiene solo {n_h} muestras (se requieren >= 2).")
        yh = y[mask]
        xh = x[mask]
        wh = stratum_weights[h]
        y_bar = float(yh.mean())
        x_bar = float(xh.mean())
        y_hat += wh * y_bar
        x_hat += wh * x_bar
        dy = yh - y_bar
        dx = xh - x_bar
        sy2 = float(np.sum(dy * dy) / (n_h - 1))
        sx2 = float(np.sum(dx * dx) / (n_h - 1))
        sxy = float(np.sum(dy * dx) / (n_h - 1))
        stats_h.append((wh, n_h, sy2, sx2, sxy))

    if np.isclose(x_hat, 0.0):
        return np.nan, np.nan

    r_hat = y_hat / x_hat
    var_r = 0.0
    for wh, n_h, sy2, sx2, sxy in stats_h:
        var_r += (wh ** 2) * (sy2 + (r_hat ** 2) * sx2 - 2.0 * r_hat * sxy) / n_h
    var_r = max(0.0, var_r / (x_hat ** 2))
    return float(r_hat), float(np.sqrt(var_r))


def stehman_generalized_accuracy(
    df: pd.DataFrame,
    stratum_col: str,
    pred_col: str,
    ref_col: str,
    stratum_areas_ha: Dict[Any, float],
    class_labels: Sequence[int],
    class_names: Dict[int, str] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 20250822,
) -> Dict[str, Any]:
    """Estimadores insesgados generalizados de Stehman (2014) para muestreo estratificado.

    Permite evaluar MÚLTIPLES mapas (MS, SAR, MS+SAR) usando una ÚNICA campaña validada
    en Collect Earth Online (CEO), aun cuando los estratos de muestreo de AREA2 no
    coincidan con las clases predichas por el mapa evaluado. Cuando ``stratum_col == pred_col``,
    los estimadores y errores estándar coinciden exactamente con Olofsson et al. (2014).
    """
    cols = list(dict.fromkeys([stratum_col, pred_col, ref_col]))
    sub = df[cols].dropna().copy()
    strata = sub[stratum_col].values
    y_pred = sub[pred_col].astype(int).values
    y_ref = sub[ref_col].astype(int).values
    labels = [int(c) for c in class_labels]
    k = len(labels)

    stratum_ids = list(stratum_areas_ha.keys())
    a_tot = float(sum(stratum_areas_ha.values()))
    w_map = {h: float(stratum_areas_ha[h]) / a_tot for h in stratum_ids}
    z = _z_value(confidence)

    # Matriz estimada de proporciones de área p_ij (pred = i, ref = j)
    p_mat = np.zeros((k, k), dtype=float)
    n_mat = np.zeros((k, k), dtype=int)
    for i, ci in enumerate(labels):
        for j, cj in enumerate(labels):
            ind = ((y_pred == ci) & (y_ref == cj)).astype(float)
            n_mat[i, j] = int(ind.sum())
            val = 0.0
            for h in stratum_ids:
                m = strata == h
                val += w_map[h] * float(ind[m].mean())
            p_mat[i, j] = val

    ones = np.ones(len(sub), dtype=float)
    oa, se_oa = _stratified_ratio_estimator(
        (y_pred == y_ref).astype(float), ones, strata, stratum_ids, w_map
    )
    kappa = _kappa_from_proportions(p_mat)

    pu_arr, pu_se_arr = np.zeros(k), np.zeros(k)
    pp_arr, pp_se_arr = np.zeros(k), np.zeros(k)
    f1_arr = np.zeros(k)
    area_adj, area_se = np.zeros(k), np.zeros(k)
    area_map_est = p_mat.sum(axis=1) * a_tot

    for idx_c, c in enumerate(labels):
        ind_both = ((y_pred == c) & (y_ref == c)).astype(float)
        ind_pred = (y_pred == c).astype(float)
        ind_ref = (y_ref == c).astype(float)

        pu_arr[idx_c], pu_se_arr[idx_c] = _stratified_ratio_estimator(
            ind_both, ind_pred, strata, stratum_ids, w_map
        )
        pp_arr[idx_c], pp_se_arr[idx_c] = _stratified_ratio_estimator(
            ind_both, ind_ref, strata, stratum_ids, w_map
        )
        f1_arr[idx_c] = _safe_f1(pu_arr[idx_c], pp_arr[idx_c])

        p_ref_c, se_p_ref_c = _stratified_ratio_estimator(
            ind_ref, ones, strata, stratum_ids, w_map
        )
        area_adj[idx_c] = a_tot * p_ref_c
        area_se[idx_c] = a_tot * se_p_ref_c

    # Bootstrap estratificado (remuestreo dentro de cada estrato h) para IC 95% de F1 y Kappa
    rng = np.random.default_rng(seed)
    indices_by_h = {h: np.where(strata == h)[0] for h in stratum_ids}
    boot_f1 = np.full((n_bootstrap, k), np.nan, dtype=float)
    boot_kappa = np.full(n_bootstrap, np.nan, dtype=float)
    boot_oa = np.full(n_bootstrap, np.nan, dtype=float)

    if n_bootstrap > 0:
        label_to_idx = {c: i for i, c in enumerate(labels)}
        pred_idx = np.array([label_to_idx.get(v, -1) for v in y_pred])
        ref_idx = np.array([label_to_idx.get(v, -1) for v in y_ref])
        for b in range(n_bootstrap):
            p_b = np.zeros((k, k), dtype=float)
            for h in stratum_ids:
                idx_h = indices_by_h[h]
                drawn = rng.choice(idx_h, size=len(idx_h), replace=True)
                pi = pred_idx[drawn]
                ri = ref_idx[drawn]
                valid = (pi >= 0) & (ri >= 0)
                counts_b = np.zeros((k, k), dtype=float)
                np.add.at(counts_b, (pi[valid], ri[valid]), 1.0)
                p_b += w_map[h] * (counts_b / len(idx_h))
            row_s = p_b.sum(axis=1)
            col_s = p_b.sum(axis=0)
            diag_s = np.diag(p_b)
            u_b = np.where(row_s > 0, diag_s / row_s, np.nan)
            pr_b = np.where(col_s > 0, diag_s / col_s, np.nan)
            for j in range(k):
                boot_f1[b, j] = _safe_f1(u_b[j], pr_b[j])
            boot_kappa[b] = _kappa_from_proportions(p_b)
            boot_oa[b] = float(np.trace(p_b))

    alpha = 1.0 - confidence
    f1_lo = np.nanpercentile(boot_f1, 100 * alpha / 2.0, axis=0) if n_bootstrap > 0 else np.full(k, np.nan)
    f1_hi = np.nanpercentile(boot_f1, 100 * (1.0 - alpha / 2.0), axis=0) if n_bootstrap > 0 else np.full(k, np.nan)
    f1_se = np.nanstd(boot_f1, axis=0, ddof=1) if n_bootstrap > 1 else np.full(k, np.nan)
    kappa_se = float(np.nanstd(boot_kappa, ddof=1)) if n_bootstrap > 1 else np.nan
    kappa_lo = float(np.nanpercentile(boot_kappa, 100 * alpha / 2.0)) if n_bootstrap > 0 else np.nan
    kappa_hi = float(np.nanpercentile(boot_kappa, 100 * (1.0 - alpha / 2.0))) if n_bootstrap > 0 else np.nan

    names_axis = [class_names[c] if class_names and c in class_names else str(c) for c in labels]
    rows = []
    for j, c in enumerate(labels):
        rows.append(
            {
                "clase_id": c,
                "clase": names_axis[j],
                "area_mapa_est_ha": area_map_est[j],
                "area_ajustada_ha": area_adj[j],
                "area_SE_ha": area_se[j],
                "area_CI_half_ha": z * area_se[j],
                "area_CI_low_ha": max(0.0, area_adj[j] - z * area_se[j]),
                "area_CI_high_ha": area_adj[j] + z * area_se[j],
                "PU": pu_arr[j],
                "PU_SE": pu_se_arr[j],
                "PU_CI_half": z * pu_se_arr[j],
                "PP": pp_arr[j],
                "PP_SE": pp_se_arr[j],
                "PP_CI_half": z * pp_se_arr[j],
                "F1": f1_arr[j],
                "F1_SE": f1_se[j],
                "F1_CI_low": f1_lo[j],
                "F1_CI_high": f1_hi[j],
            }
        )

    p_df = pd.DataFrame(p_mat, index=names_axis, columns=names_axis)
    p_df.index.name = "Predicho"
    p_df.columns.name = "Referencia (CEO)"

    c_df = pd.DataFrame(n_mat, index=names_axis, columns=names_axis)
    c_df.index.name = "Predicho"
    c_df.columns.name = "Referencia (CEO)"

    return {
        "metodo": "Stehman (2014) / Olofsson et al. (2014)",
        "n_muestras": int(len(sub)),
        "area_total_ha": a_tot,
        "OA": oa,
        "OA_SE": se_oa,
        "OA_CI_half": z * se_oa,
        "OA_CI_low": max(0.0, oa - z * se_oa),
        "OA_CI_high": min(1.0, oa + z * se_oa),
        "Kappa": kappa,
        "Kappa_SE": kappa_se,
        "Kappa_CI_low": kappa_lo,
        "Kappa_CI_high": kappa_hi,
        "per_class": pd.DataFrame(rows),
        "matrix_proportions": p_df,
        "matrix_counts": c_df,
        "bootstrap_f1": boot_f1,
        "bootstrap_kappa": boot_kappa,
    }


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    binary_target_class: int | None = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de McNemar para comparar dos clasificaciones pareadas sobre la misma muestra.

    Si ``binary_target_class`` se especifica (ej. ``1`` para Mangle), evalúa la
    discriminación binaria (Mangle vs. No-Mangle); si es ``None``, evalúa exactitud multiclase.
    """
    yt = np.asarray(y_true, dtype=int)
    ya = np.asarray(y_pred_a, dtype=int)
    yb = np.asarray(y_pred_b, dtype=int)

    if binary_target_class is not None:
        yt_eval = (yt == binary_target_class)
        ca = (ya == binary_target_class) == yt_eval
        cb = (yb == binary_target_class) == yt_eval
    else:
        ca = ya == yt
        cb = yb == yt

    both_ok = int(np.sum(ca & cb))
    a_only = int(np.sum(ca & ~cb))  # b: A acierta, B falla
    b_only = int(np.sum(~ca & cb))  # c: B acierta, A falla
    both_err = int(np.sum(~ca & ~cb))

    discordant = a_only + b_only
    if discordant == 0:
        chi2_cc = 0.0
        p_chi2 = 1.0
        p_exact = 1.0
    else:
        chi2_cc = float(((abs(a_only - b_only) - 1.0) ** 2) / discordant) if abs(a_only - b_only) > 0 else 0.0
        p_chi2 = float(1.0 - stats.chi2.cdf(chi2_cc, df=1))
        # Prueba binomial exacta bilateral (más confiable cuando b + c < 25)
        p_exact = float(stats.binomtest(min(a_only, b_only), n=discordant, p=0.5, alternative="two-sided").pvalue)

    p_recom = p_exact if discordant < 25 else p_chi2
    return {
        "ambos_aciertan": both_ok,
        "solo_A_acierta_b": a_only,
        "solo_B_acierta_c": b_only,
        "ambos_fallan": both_err,
        "discordantes": discordant,
        "chi2_continuidad": chi2_cc,
        "p_value_chi2": p_chi2,
        "p_value_exacto": p_exact,
        "p_value_recomendado": p_recom,
        "rechaza_H0": bool(p_recom < alpha),
    }


def stratified_paired_z_test(
    df: pd.DataFrame,
    stratum_col: str,
    pred_a_col: str,
    pred_b_col: str,
    ref_col: str,
    stratum_areas_ha: Dict[Any, float],
    binary_target_class: int | None = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba Z pareada ponderada por diseño muestral estratificado (ΔOA = OA_A - OA_B)."""
    cols = list(dict.fromkeys([stratum_col, pred_a_col, pred_b_col, ref_col]))
    sub = df[cols].dropna()
    strata = sub[stratum_col].values
    yt = sub[ref_col].astype(int).values
    ya = sub[pred_a_col].astype(int).values
    yb = sub[pred_b_col].astype(int).values

    if binary_target_class is not None:
        yt_eval = yt == binary_target_class
        ca = ((ya == binary_target_class) == yt_eval).astype(float)
        cb = ((yb == binary_target_class) == yt_eval).astype(float)
    else:
        ca = (ya == yt).astype(float)
        cb = (yb == yt).astype(float)

    diff = ca - cb
    ones = np.ones(len(sub), dtype=float)
    stratum_ids = list(stratum_areas_ha.keys())
    a_tot = float(sum(stratum_areas_ha.values()))
    w_map = {h: float(stratum_areas_ha[h]) / a_tot for h in stratum_ids}

    delta, se_delta = _stratified_ratio_estimator(diff, ones, strata, stratum_ids, w_map)
    if np.isnan(se_delta) or np.isclose(se_delta, 0.0):
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = float(delta / se_delta)
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    return {
        "delta_OA_A_menos_B": delta,
        "SE_delta": se_delta,
        "Z_stat": z_stat,
        "p_value": p_val,
        "rechaza_H0": bool(p_val < alpha),
    }


def kappa_z_test(
    kappa_a: float,
    se_a: float,
    kappa_b: float,
    se_b: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba Z de Congalton & Green para diferencia entre dos coeficientes Kappa."""
    denom = float(np.sqrt(se_a ** 2 + se_b ** 2))
    if np.isnan(denom) or np.isclose(denom, 0.0):
        return {"delta_kappa": kappa_a - kappa_b, "Z_stat": np.nan, "p_value": np.nan, "rechaza_H0": False}
    z_stat = float((kappa_a - kappa_b) / denom)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    return {
        "delta_kappa": float(kappa_a - kappa_b),
        "Z_stat": z_stat,
        "p_value": p_val,
        "rechaza_H0": bool(p_val < alpha),
    }
