# TFI Neurociencias y Ciencia de Datos · BCI de imaginería motora

**Pregunta central:** ¿cuánto pierde un clasificador de imaginería motora (mano izquierda vs. derecha)
al usarse en una persona nueva, qué decisiones del pipeline influyen en esa pérdida, y cuántos
ensayos de calibración hacen falta para recuperarla?

## Datos

[EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (PhysioNet):
109 sujetos, 64 canales EEG, 160 Hz. Se usan los runs 4, 8 y 12 (imaginar abrir/cerrar el puño
izquierdo, anotación `T1`, o derecho, `T2`). Los datos **no se versionan**: se descargan solos
a `data/` la primera vez que se cargan.

Se excluyen los sujetos **88, 92 y 100**, grabados a 128 Hz en lugar de 160 Hz; quedan 106.
Los demás tienen entre 36 y 57 ensayos (45 en 96 de ellos). El detalle está en `resultados/sujetos.csv`.

Licencia: Open Data Commons Attribution v1.0. Citar (según la página del dataset):
- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (v1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P
- Schalk et al. (2004), *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System*, IEEE TBME 51(6), 1034–1043.
- Pollard et al. (2026), *PhysioNet as a global platform for biomedical research*, Nature Health. https://doi.org/10.1038/s44360-026-00096-z

## Instalación

```bash
python -m venv .venv            # Python 3.13
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

> **Para ver los resultados no hace falta correr nada.** Los CSV, las figuras y los notebooks
> con sus salidas ya están en el repo (`resultados/` y `notebooks/`). Los notebooks 04 y 05 solo
> leen los CSV, así que se pueden volver a ejecutar en segundos. Solo hay que correr
> `experimentos/etapa4.py` (~3 h) o `etapa5.py` (~45 min) si se cambia algo de esas etapas. Los
> notebooks 01 y 02 descargan el dataset de PhysioNet la primera vez que se ejecutan.

## Estructura

| Ruta | Contenido |
|---|---|
| `bci/datos.py` | Carga y segmentación común a todo el grupo (`cargar_raw`, `epocas`). |
| `bci/pipelines.py` | Dataset completo en caché (`cargar_dataset`), clasificadores (`crear_pipeline`) y validación dentro del sujeto / leave-subject-out. |
| `verificar_sujetos.py` | Descarga los 109 sujetos y resume frecuencia de muestreo y ensayos (`resultados/sujetos.csv`). |
| `notebooks/01_exploracion.ipynb` | Verifica el ERD en mu/beta sobre C3/C4 y su variabilidad entre sujetos. |
| `notebooks/02_intra_vs_entre_sujetos.ipynb` | Etapas 2 y 3: CSP + LDA dentro del sujeto vs. leave-subject-out (H1). |
| `experimentos/etapa4.py` | Etapa 4: 10 configuraciones (modelo, banda, ventana, alineación euclídea) en los dos esquemas (~3 h). Resultados en `resultados/04_resultados.csv`. |
| `experimentos/etapa5.py` | Etapa 5: curva de calibración con 0 a 30 ensayos del sujeto nuevo (~45 min). Resultados en `resultados/05_calibracion.csv`. |
| `experimentos/figura_erd.py`, `figura_etapa4.py`, `figura_etapa5.py` | Figuras del informe: ERD en mu (1 min, 20 sujetos) y etapas 4 y 5 a partir de los CSV (segundos). |
| `notebooks/04_configuraciones.ipynb` | Análisis de la etapa 4 (H2): tablas, pruebas de Wilcoxon con Holm, figuras e ITR. Lo genera `experimentos/crear_nb04.py`. |
| `notebooks/05_calibracion.ipynb` | Análisis de la etapa 5 (H3): curva de calibración y pruebas. Lo genera `experimentos/crear_nb05.py`. |

**Regla del grupo:** todo notebook carga los datos con `bci.datos`; si hace falta otro
preprocesamiento, se agrega como parámetro ahí, no se reescribe en el notebook.
