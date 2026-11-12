"""Genera notebooks/05_calibracion.ipynb (solo analiza resultados/05_calibracion.csv; no reentrena).

Uso (desde la raíz del repo):  python experimentos/crear_nb05.py notebooks/05_calibracion.ipynb experimentos/lectura_05.md
Después:  jupyter nbconvert --to notebook --execute --inplace notebooks/05_calibracion.ipynb
"""

import sys

import nbformat as nbf

lectura = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else "## Lectura de los resultados\n\n_(pendiente)_"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
md("""# 05 · Curva de calibración: ¿cuántos ensayos del sujeto nuevo hacen falta, y para qué?

**H3:** con 10 a 20 ensayos de calibración se recupera buena parte de la diferencia.

Para cada uno de los 106 sujetos que se deja afuera se toman **k** ensayos de calibración (k = 0, 5, 10, 20, 30, estratificados por clase) y se evalúa con el resto de sus ensayos. Cada k se repite 10 veces con distintos ensayos de calibración. Configuración base del notebook 02 (CSP + LDA, 8–30 Hz, 0,5–2,5 s).

| Estrategia | Qué usa del sujeto nuevo |
|---|---|
| General, sin ajustar | Nada: modelo entrenado con los otros 105 sujetos |
| General + umbral | Los k ensayos, solo para recentrar el umbral de decisión (punto medio entre los puntajes medios de cada clase). No cambia el AUC |
| General alineado + umbral | Modelo entrenado con sujetos alineados (alineación euclídea); el sujeto nuevo se alinea con la covarianza de sus k ensayos y se recentra el umbral |
| Personal | Un CSP + LDA entrenado solo con sus k ensayos (k ≥ 10) |

A diferencia del notebook 04, la alineación del sujeto nuevo usa **solo** sus k ensayos de calibración: es la versión realista.

Los resultados se calculan con `experimentos/etapa5.py` (unos 45 minutos). Este notebook solo los analiza."""),
code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
from IPython.display import Image
from scipy.stats import wilcoxon

from bci.pipelines import itr_bits_por_minuto, umbral_azar

RESULTADOS = Path("..") / "resultados"
NOMBRE = {"general": "General, sin ajustar", "general_umbral": "General + umbral",
          "alineado_umbral": "General alineado + umbral", "personal": "Personal"}

df = pd.read_csv(RESULTADOS / "05_calibracion.csv")
# Primero se promedian las 10 repeticiones de cada sujeto, para que todos los sujetos pesen igual.
suj = df.groupby(["sujeto", "k", "estrategia"])[["balanced_accuracy", "auc", "n_prueba"]].mean().reset_index()
suj["sobre_azar"] = suj["balanced_accuracy"] >= suj["n_prueba"].round().astype(int).map(umbral_azar)
suj["itr"] = itr_bits_por_minuto(suj["balanced_accuracy"], 2.5)  # 2,5 s por decisión (final de la ventana)
print(suj.groupby(["k", "estrategia"]).size().unstack())
"""),
md("## Tabla resumen\n\nMedias sobre los 106 sujetos. *Sobre el azar*: sujetos cuya balanced accuracy supera el umbral binomial (p < 0,05) para su cantidad de ensayos de prueba."),
code("""tabla = suj.groupby(["k", "estrategia"]).agg(
    BA=("balanced_accuracy", "mean"), AUC=("auc", "mean"), ITR=("itr", "mean"), sobre_azar=("sobre_azar", "sum"))
tabla = tabla.rename(index=NOMBRE, level="estrategia")
tabla.to_csv(RESULTADOS / "05_tabla_resumen.csv")
tabla.round(3)
"""),
md("""## Pruebas estadísticas

Cada estrategia contra el modelo general sin ajustar **con el mismo k** (mismos ensayos de prueba), Wilcoxon pareado por sujeto. También: ¿sigue mejorando el modelo alineado al agregar ensayos?"""),
code("""ancho = suj.pivot_table(index=["sujeto", "k"], columns="estrategia", values="balanced_accuracy")
filas = []
for k in [5, 10, 20, 30]:
    a = ancho.xs(k, level="k")
    for e in ["general_umbral", "alineado_umbral", "personal"]:
        if a[e].isna().all():
            continue
        filas.append({"k": k, "estrategia": NOMBRE[e], "BA": a[e].mean(), "BA general": a["general"].mean(),
                      "diferencia": (a[e] - a["general"]).mean(), "p Wilcoxon": wilcoxon(a[e], a["general"]).pvalue,
                      "sujetos que mejoran": int((a[e] > a["general"]).sum())})
pruebas = pd.DataFrame(filas)
pruebas.to_csv(RESULTADOS / "05_pruebas.csv", index=False)
with pd.option_context("display.float_format", "{:.3g}".format):
    display(pruebas)

alin = ancho["alineado_umbral"].unstack("k")
for k1, k2 in [(5, 10), (10, 20), (20, 30)]:
    print(f"Alineado + umbral, k={k1} -> k={k2}: {alin[k1].mean():.3f} -> {alin[k2].mean():.3f}, "
          f"p = {wilcoxon(alin[k1], alin[k2]).pvalue:.2g}")
"""),
md("""## Figura · Curva de calibración

La genera `experimentos/figura_etapa5.py` (es la misma que va en el informe). La línea punteada es el modelo personal del notebook 02, entrenado con unos 36 ensayos por validación cruzada."""),
code("""from experimentos.figura_etapa5 import main as figura_calibracion

figura_calibracion()
Image(RESULTADOS / "05_calibracion.png", width=800)
"""),
md("""## ¿A quién ayuda la calibración?

Mejora del modelo alineado con 10 ensayos respecto del general sin ajustar, según qué tan bien le iba a cada sujeto con el modelo general."""),
code("""a10 = ancho.xs(10, level="k")
grupos = pd.qcut(a10["general"], 3, labels=["tercio bajo", "tercio medio", "tercio alto"])
(a10["alineado_umbral"] - a10["general"]).groupby(grupos, observed=True).agg(["mean", "median", "count"]).round(3)
"""),
md(lectura),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
