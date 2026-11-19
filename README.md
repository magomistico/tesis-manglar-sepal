# Evaluación de un modelo geoespacial S1 + S2 para la estimación de cobertura manglar en Guatemala (SEPAL.io)

Pipeline reproducible en **Python + Jupyter Notebooks** diseñado para ejecutarse en máquinas virtuales de **[SEPAL.io](https://sepal.io/)** y **Google Earth Engine (GEE)**, correspondiente a la tesis de Maestría en Ciencias en Geomática (USAC):

> **Título:** *Evaluación de un modelo geoespacial de imágenes Sentinel-1 y Sentinel-2 para la estimación de cobertura manglar*  
> **Autor:** Marlon Ronaldo Vásquez Reyes  
> **Áreas de estudio:** Costa del Pacífico (`EPSG:32615`) y Costa del Caribe (`EPSG:32616`), Guatemala.

---

## Arquitectura del flujo de trabajo

El proyecto está organizado en 6 notebooks secuenciales alineados a los Objetivos Específicos (OE1, OE2 y OE3) del Protocolo FINAL:

| Notebook | Objetivo / Etapa | Salidas principales | Instancia SEPAL recomendada |
| :--- | :--- | :--- | :--- |
| [`00_setup_entorno_sepal.ipynb`](notebooks/00_setup_entorno_sepal.ipynb) | Verificación de entorno, autenticación GEE y auditoría de assets (`AOI_PACIFICO`, `AOI_CARIBE`, `DinMangle_2020_2022_V23`) | Diagnóstico de acceso a assets y espacio en disco | `t2` (muy económica) |
| [`01_stack_biestacional_S1_S2.ipynb`](notebooks/01_stack_biestacional_S1_S2.ipynb) | Construcción desde cero del **Stack Biestacional S1 + S2** (24 bandas a 10 m UTM + 6 bandas QA, doble anillo 100 m + 60 m) | Assets en GEE: Stack 24b, QA 6b, Inventario S1 y Metadatos CSV | `t2` / `m2` (cómputo en GEE) |
| [`02_colector_puntos_control.ipynb`](notebooks/02_colector_puntos_control.ipynb) | Colector interactivo de puntos de control en mapa sobre `DinMangle_2020_2022_V23` + stack, deduplicación 10 m y split preliminar 70/30 | `muestras_control_raw_<region>.gpkg`, `muestras_limpias_<region>_70_30.gpkg` y Asset GEE | `m2` |
| [`03_pca_seleccion_variables_OE1.ipynb`](notebooks/03_pca_seleccion_variables_OE1.ipynb) | **OE1:** Análisis de Componentes Principales (Kaiser $\lambda \ge 1$), contraste Seca vs. Lluvia y separabilidad multiclase (ANOVA F, $d$ de Cohen) | `bandas_seleccionadas_pca_<region>.json`, `OE1_PCA_Resultados_<region>.xlsx`, 3 figuras 300 DPI | `m2` / `m4` |
| [`04_clasificacion_RF_mapas_OE2.ipynb`](notebooks/04_clasificacion_RF_mapas_OE2.ipynb) | **OE2:** Optimización de Random Forest (4 modelos: `MS`, `SAR`, `MS_SAR_PCA`, `MS_SAR_FULL`), diagnóstico preliminar 70/30 (OA, Kappa, **F1**), clasificación a 10 m y mapa de estratos para **AREA2** | Mapas en GEE Assets + GeoTIFF COG en Drive, `areas_estratos_<region>.json`, `OE2_superficies_mapeadas_<region>.csv` | `m4` / `m8` (tuning CV) |
| [`05_validacion_olofsson_CEO_OE3.ipynb`](notebooks/05_validacion_olofsson_CEO_OE3.ipynb) | **OE3 & Hipótesis:** Ingesta de validación **Collect Earth Online (CEO)** (Sentinel-2 + Planet), estimadores insesgados **Olofsson (2014) / Stehman (2014)**, **F1-Score ± IC 95%**, área ajustada (ha) y pruebas **McNemar / Z** | `OE3_Validacion_Olofsson_Hipotesis_<region>.xlsx` y figuras finales 300 DPI | `m2` |

---

## Estructura del repositorio

```text
tesis-manglar-sepal/
├── config.yaml                          # Parámetros maestros (GEE, CRS, fechas, bandas, clases, RF, Olofsson)
├── requirements.txt                     # Dependencias Python para SEPAL.io
├── src/
│   ├── config_utils.py                  # Carga de configuración, rutas e inicialización GEE
│   ├── s2_preproc.py                    # Preprocesamiento óptico Sentinel-2 (Cloud Score+ / s2cloudless)
│   ├── s1_ard.py                        # Cadena Sentinel-1 ARD en Python puro (Border Noise, Refined Lee, Terrain Flattening)
│   ├── stack_builder.py                 # Ensamblaje de Stack 24b + QA 6b, diagnóstico y exportación
│   ├── metrics.py                       # Métricas 70/30, Olofsson (2014), Stehman (2014), F1 bootstrap, McNemar y Z
│   └── plotting.py                      # Figuras académicas de publicación (300 DPI)
├── notebooks/                           # Notebooks 00 a 05 (flujo completo)
├── outputs/                             # Salidas locales generadas (tablas, figuras, muestras, modelos, mapas)
└── tests/
    └── test_metrics_and_config.py       # Pruebas unitarias de métricas estadísticas y configuración
```

---

## Inicio rápido en SEPAL.io

1. En SEPAL.io, abre una terminal (`Terminal`) en una instancia económica (`t2` o `m2`).
2. Clona este repositorio **dentro de tu almacenamiento persistente** (`~/`), **nunca** en `/tmp` (el SSD `/tmp` es efímero y se borra al apagar la instancia):

```bash
cd ~
git clone https://github.com/magomistico/tesis-manglar-sepal.git
cd tesis-manglar-sepal
pip install -r requirements.txt
```

3. Verifica y ajusta en [`config.yaml`](config.yaml) las rutas exactas de tus assets en GEE:
   - `gee.project` y `gee.asset_root` (por defecto `manglev1`)
   - `assets.aoi.PACIFICO` y `assets.aoi.CARIBE`
   - `assets.mangle_guia` (`projects/tesismangle/assets/DinMangle_2020_2022_V23`)
4. Abre **JupyterLab** en SEPAL y comienza ejecutando `notebooks/00_setup_entorno_sepal.ipynb`.

> **Recomendación de presupuesto SEPAL:** Apaga tu instancia al terminar cada sesión de trabajo; tus notebooks y archivos en `~/tesis-manglar-sepal` permanecerán guardados de forma persistente. Usa `t2`/`m2` para preparación, muestreo y validación, y sube a `m4`/`m8` únicamente durante la búsqueda de hiperparámetros de Random Forest en el Notebook 04.

---

## Flujo de validación independiente: AREA2 (GEE) → Collect Earth Online (CEO) → Notebook 05

Para evaluar de manera justa e insesgada los 4 modelos (`MS`, `SAR`, `MS_SAR_PCA` y `MS_SAR_FULL`) usando una **única campaña de fotointerpretación**:

1. **Notebook 04** exporta a GEE Assets los 4 mapas clasificados en UTM 10 m (`EPSG:32615` / `EPSG:32616`) y un mapa de 6 estratos de **Acuerdo/Desacuerdo en Manglar** (`mapa_estratos_area2_<region>_v1_sepal`), guardando las hectáreas exactas en `outputs/tablas/areas_estratos_<region>.json`.
2. **AREA2 (GEE):** Carga el mapa de estratos (`mapa_estratos_area2_...` o `mapa_ms_sar_pca_...`), calcula el tamaño de muestra estratificado óptimo (Olofsson et al., 2014) y exporta los puntos muestrales en **`EPSG:4326`** (`lon`, `lat`, `stratum`).
3. **Collect Earth Online (CEO):** Sube el CSV de AREA2 como proyecto en CEO y valida visualmente cada punto utilizando mosaicos Sentinel-2 y Planet (NICFI / alta resolución), asignando la clase de referencia (`1`–`5`).
4. **Notebook 05:** Carga el CSV exportado desde CEO en `outputs/muestras/ceo_validacion_<region>.csv`, cruza automáticamente las coordenadas con los 4 mapas en GEE y calcula:
   - Precisión Global (**OA**), Precisión de Usuario (**PU**), Precisión de Productor (**PP**), **F1-Score por clase** y **Índice Kappa** con **IC 95%** (estimadores generalizados de Stehman 2014 / Olofsson 2014 + bootstrap estratificado).
   - **Superficie ajustada insesgada de manglar (ha ± IC 95%)** comparada con la línea base MARN (2013).
   - Contraste formal de hipótesis ($H_0$ vs. $H_1$): **Prueba de McNemar** (multiclase y binaria Mangle vs. No-Mangle), **Prueba Z pareada ponderada por diseño** y **Prueba Z de Kappa**.

---

## Decisiones metodológicas y controles de calidad

- **Sistemas de referencia (CRS):** Los stacks raster y el cálculo de superficies operan en la grilla métrica nativa de 10 m por litoral (`EPSG:32615` Pacífico y `EPSG:32616` Caribe) para evitar distorsiones de área y escala; todas las geometrías vectoriales, puntos de control dibujados, muestreo AREA2 y validación CEO se almacenan e intercambian en **`EPSG:4326`**.
- **Sentinel-1 ARD en Python puro:** Implementación autocontenida basada en Mullissa et al. (2021) y Vollrath et al. (2020) con DEM costero protegido (`.unmask(0)`), filtro direccional Refined Lee 7×7 en potencia lineal y piso mínimo `1e-6` (-60 dB).
- **Sentinel-2 libre de falsas sombras en manglar:** Enmascaramiento por defecto con `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` (`cs_cdf >= 0.60`), con opción configurable a `s2cloudless` + protección MNDWI/SCL en `config.yaml`.
- **Seguridad en exportaciones:** Todas las celdas de lanzamiento de tareas en GEE están protegidas mediante banderas `EJECUTAR_EXPORTS = False` para inspeccionar primero las estadísticas y mapas de control.
- **Corrección de fórmula de Precisión de Productor:** Se implementa la fórmula estándar $PP = VP / (VP + FN)$ (corrigiendo la errata tipográfica $VP - FN$ del protocolo).

---

## Verificación de pruebas automatizadas

```bash
pytest -v
```
