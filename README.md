# Evaluación de un modelo geoespacial S1 + S2 para la estimación de cobertura manglar en Guatemala (SEPAL.io)

Pipeline reproducible en **Python + Jupyter Notebooks** diseñado para ejecutarse **100% en las máquinas virtuales de [SEPAL.io](https://sepal.io/)** (sin depender de exportaciones lentas de Google Earth Engine ni Google Drive), correspondiente a la tesis de Maestría en Ciencias en Geomática (USAC):

> **Título:** *Evaluación de un modelo geoespacial de imágenes Sentinel-1 y Sentinel-2 para la estimación de cobertura manglar*  
> **Autor:** Marlon Ronaldo Vásquez Reyes  
> **Áreas de estudio:** Costa del Pacífico (`EPSG:32615`) y Costa del Caribe (`EPSG:32616`), Guatemala.

---

## Arquitectura 100% SEPAL.io (Cero exportaciones pesadas de GEE)

A partir del Notebook 01 todo el procesamiento pesado corre directamente sobre los núcleos CPU y el SSD `/tmp` de tu VM en SEPAL (`m8d` o `m16d`):

- **Sentinel-2 L2A (`sentinel-2-c1-l2a`):** lectura por ventanas (COGs) directa desde el catálogo público **AWS Earth Search STAC** (sin credenciales), con verificación física anti-doble-offset y protección espectral de manglar inundado mediante `SCL == 6` + `MNDWI >= 0.10`.
- **Sentinel-1 ARD:** fuente predeterminada `OPERA_RTC_S1` vía **NASA ASF** (COGs oficiales NASA/JPL `gamma0` con corrección de terreno Copernicus GLO-30 DEM), filtro direccional **Refined Lee 7×7 en potencia lineal** vectorizado en NumPy/SciPy, piso seguro `-60 dB` (`1e-6`) y mediana estacional local. Soporta también modo opcional `GRD_SNAP` si ESA SNAP `gpt` está instalado en la VM.
- **Uso puntual de GEE en Notebook 00:** descarga **una sola vez** los dos polígonos `AOI_PACIFICO` / `AOI_CARIBE` (`outputs/aoi/`) y el recorte ligero `uint8` de `DinMangle_2020_2022_V23` (`outputs/guia/`). Opcionalmente se conserva `computo.modo: "gee"` en `config.yaml` solo como respaldo.

---

## Flujo de notebooks

| Notebook | Objetivo / Etapa | Salidas principales locales | Instancia SEPAL recomendada |
| :--- | :--- | :--- | :--- |
| [`00_setup_entorno_sepal.ipynb`](notebooks/00_setup_entorno_sepal.ipynb) | Verificación de librerías, detección de SNAP `gpt`, credenciales **NASA Earthdata** (`~/.netrc`) y descarga única de AOIs + `DinMangle` guía | `outputs/aoi/AOI_*.gpkg`, `outputs/guia/DinMangle_*.tif` | `t2` (\$0.04/h) |
| [`01_stack_biestacional_S1_S2.ipynb`](notebooks/01_stack_biestacional_S1_S2.ipynb) | Construcción 100% local en SEPAL del **Stack Biestacional S1 + S2** (24 bandas a 10 m UTM + 6 bandas QA, doble anillo 100 m + 60 m) | `outputs/stack/stack_biestacional_*.tif` (24b `float32`) y `qa_stack_*.tif` (6b `uint16`) | `m8d` (8 CPU / 32 GB / 474 GB SSD, \$0.53/h) o `m16d` |
| [`02_colector_puntos_control.ipynb`](notebooks/02_colector_puntos_control.ipynb) | Colector interactivo de puntos de control en mapa sobre stack local + `DinMangle_2020_2022_V23`, muestreo en RAM (`rasterio.sample`), deduplicación 10 m y split 70/30 | `muestras_control_raw_*.gpkg`, `muestras_limpias_*_70_30_con_bandas.csv` y `.gpkg` | `t2` / `m4` |
| [`03_pca_seleccion_variables_OE1.ipynb`](notebooks/03_pca_seleccion_variables_OE1.ipynb) | **OE1:** Análisis de Componentes Principales (criterio de Kaiser $\lambda \ge 1$), contraste Seca vs. Lluvia y separabilidad multiclase (ANOVA F, $d$ de Cohen) | `bandas_seleccionadas_pca_*.json`, `OE1_PCA_Resultados_*.xlsx`, figuras 300 DPI | `t2` / `m4` |
| [`04_clasificacion_RF_mapas_OE2.ipynb`](notebooks/04_clasificacion_RF_mapas_OE2.ipynb) | **OE2:** Optimización Random Forest (4 modelos: `MS`, `SAR`, `MS_SAR_PCA`, `MS_SAR_FULL`), diagnóstico 70/30 (OA, Kappa, **F1**), clasificación a 10 m en GeoTIFF por bloques, mapa de 6 estratos de Acuerdo/Desacuerdo y diseño muestral AREA2/Olofsson | GeoTIFFs `uint8` en `outputs/mapas/`, `areas_estratos_*.json`, `diseno_muestral_area2_para_CEO_*.csv` | `m8d` / `m16d` |
| [`05_validacion_olofsson_CEO_OE3.ipynb`](notebooks/05_validacion_olofsson_CEO_OE3.ipynb) | **OE3 & Hipótesis:** Cruce en 1 s de coordenadas de **Collect Earth Online (CEO)** (Sentinel-2 + Planet) contra los 4 GeoTIFFs locales, estimadores insesgados **Olofsson (2014) / Stehman (2014)**, **F1-Score ± IC 95%**, superficie ajustada (ha) y pruebas **McNemar / Z** | `ceo_cruzado_4modelos_*.csv`, `OE3_Validacion_Olofsson_Hipotesis_*.xlsx` y figuras finales 300 DPI | `t2` |

---

## Estructura del repositorio

```text
tesis-manglar-sepal/
├── config.yaml                          # Parámetros maestros (computo.modo, STAC S2, ASF S1, CRS, clases, RF, Olofsson)
├── requirements.txt                     # Dependencias Python para SEPAL.io
├── src/
│   ├── config_utils.py                  # Carga de config.yaml, rutas locales y cálculo dinámico de max_workers
│   ├── local_aoi.py                     # Descarga puntual GEE -> GPKG/GeoTIFF local y grilla UTM exacta 10 m
│   ├── local_s2.py                      # Sentinel-2 C1 L2A vía AWS Earth Search STAC + máscara S2/MNDWI/SCL
│   ├── local_s1.py                      # Sentinel-1 OPERA RTC-S1 (ASF) + Refined Lee 7×7 vectorizado + opción SNAP gpt
│   ├── local_stack.py                   # Ensamblaje chunked de Stack 24b + QA 6b y visualización local EPSG:3857 en mapa
│   ├── local_rf.py                      # Extracción rápida, predicción RF por bloques a GeoTIFF y muestreo Olofsson/CEO
│   ├── metrics.py                       # Métricas 70/30, Olofsson (2014), Stehman (2014), F1 bootstrap, McNemar y Z
│   ├── plotting.py                      # Figuras académicas de publicación (300 DPI)
│   ├── s2_preproc.py                    # [Respaldo GEE opcional] Preprocesamiento óptico Sentinel-2
│   ├── s1_ard.py                        # [Respaldo GEE opcional] Cadena Sentinel-1 ARD en GEE
│   └── stack_builder.py                 # [Respaldo GEE opcional] Stack Builder en servidores GEE
├── notebooks/                           # Notebooks 00 a 05 (flujo completo)
├── outputs/                             # Salidas locales generadas (ignoradas en git; persistentes en ~/tesis-manglar-sepal)
└── tests/
    ├── test_metrics_and_config.py       # Pruebas estadísticas (Olofsson, Stehman, F1, McNemar/Z), config, gráficos y sintaxis
    └── test_local_pipeline.py           # Pruebas unitarias y end-to-end del pipeline raster local en SEPAL
```

---

## Inicio rápido y actualización en SEPAL.io

1. En SEPAL.io, inicia una instancia económica `t2` y abre la **Terminal** dentro de tu almacenamiento persistente (`~/`, **nunca** clones en `/tmp` porque es efímero al apagar la máquina).
   - **Si ya habías clonado el repositorio anteriormente:**

```bash
cd ~/tesis-manglar-sepal
git pull origin main
pip install -r requirements.txt
```

   - **Si es la primera vez que clonas el proyecto:**

```bash
cd ~
git clone https://github.com/magomistico/tesis-manglar-sepal.git
cd tesis-manglar-sepal
pip install -r requirements.txt
```

2. Crea una cuenta gratuita en **[NASA Earthdata Login](https://urs.earthdata.nasa.gov/)** y, dentro de *Earthdata Profile → Applications*, autoriza **`Alaska Satellite Facility Data Access`** para poder descargar bursts de Sentinel-1 (`OPERA_RTC_S1`) desde el Notebook 01.
3. Abre **JupyterLab**, ejecuta `notebooks/00_setup_entorno_sepal.ipynb` (descarga única de `AOI_PACIFICO`, `AOI_CARIBE` y `DinMangle_2020_2022_V23`, más configuración opcional de `~/.netrc` con `ACTUALIZAR_CREDENCIALES_EARTHDATA = True`).
4. Para procesar el stack (**Notebook 01**) y entrenar/mapear Random Forest (**Notebook 04**), reinicia la instancia SEPAL en **`m8d`** (o **`m16d`**). Prueba primero con `REGION = "CARIBE"` (región más compacta) y luego ejecuta `REGION = "PACIFICO"`.

---

## Flujo de validación independiente: SEPAL (Olofsson/AREA2) → Collect Earth Online (CEO) → Notebook 05

Para evaluar los 4 modelos (`MS`, `SAR`, `MS_SAR_PCA` y `MS_SAR_FULL`) con una **única campaña ciega de fotointerpretación**:

1. **Notebook 04** genera los 4 GeoTIFFs clasificados `uint8` a 10 m (`EPSG:32615` / `EPSG:32616`), construye el mapa de 6 estratos de **Acuerdo/Desacuerdo en Mangle** (`mapa_estratos_area2_<region>_<version>.tif`), calcula las hectáreas exactas en `outputs/tablas/areas_estratos_<region>.json` y exporta directamente en SEPAL el CSV estratificado (`Olofsson et al., 2014, Eq. 13`) `outputs/muestras/diseno_muestral_area2_para_CEO_<region>.csv` (`PLOTID, LON, LAT, stratum` en `EPSG:4326`). *(También puedes usar la aplicación AREA2 de GEE subiendo los mapas si lo prefieres.)*
2. **Collect Earth Online (CEO):** Sube el CSV de diseño como proyecto en CEO e interpreta cada punto usando mosaicos Sentinel-2 y Planet de alta resolución, asignando la clase de referencia (`1`–`5`).
3. **Notebook 05:** Lee `outputs/muestras/ceo_validacion_<region>.csv`, cruza en segundos las coordenadas contra los 4 GeoTIFFs locales (`cross_points_with_local_maps`) y genera:
   - **OA**, **PU**, **PP**, **F1-Score por clase** e **Índice Kappa** con **IC 95%** (estimadores generalizados Stehman 2014 / Olofsson 2014 + bootstrap estratificado).
   - **Superficie ajustada insesgada de manglar (ha ± IC 95%)** comparada contra la línea base MARN (2013).
   - Pruebas formales de hipótesis ($H_0$ vs. $H_1$): **McNemar** (multiclase y binaria Mangle vs. No-Mangle), **Prueba Z pareada ponderada por diseño** y **Prueba Z de Kappa**.

---

## Decisiones metodológicas y controles de seguridad

- **Sistemas de referencia (CRS):** Todos los rasters (stacks, máscaras y mapas finales) operan en la grilla métrica UTM exacta de 10 m (`EPSG:32615` Pacífico y `EPSG:32616` Caribe) para garantizar superficies exactas en hectáreas; las geometrías vectoriales, puntos de control y diseño/validación CEO se almacenan e intercambian en **`EPSG:4326`**.
- **Banderas de seguridad en notebooks:** Las acciones pesadas o sensibles están protegidas por defecto mediante `EJECUTAR_PROCESAMIENTO_LOCAL = False`, `GENERAR_CANDIDATOS_MANGLE = False` y `ACTUALIZAR_CREDENCIALES_EARTHDATA = False`.
- **Corrección de Precisión de Productor:** Se implementa la fórmula estándar $PP = VP / (VP + FN)$ (corrigiendo la errata tipográfica $VP - FN$ del protocolo).

---

## Verificación de pruebas automatizadas

```bash
pytest -v
```
