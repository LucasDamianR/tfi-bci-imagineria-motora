import sys
import nbformat as nbf

lectura = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else "## Lectura de los resultados\n\n_(pendiente)_"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
md("""# 04 · Comparación de configuraciones del pipeline

**Preguntas:** ¿qué decisiones del pipeline cambian el desempeño dentro del sujeto y entre sujetos? En particular, **H2**: ¿CSP generaliza mejor que la potencia por banda?

Cada configuración cambia **un solo factor** respecto de la base del notebook 02 (CSP + LDA, 8–30 Hz, ventana 0,5–2,5 s, referencia promedio):

| Factor | Niveles |
|---|---|
| Modelo | potencia por banda (mu y beta, 21 canales motores) + LDA · CSP + LDA · CSP + SVM (RBF) · CSP + Random Forest |
| Banda | mu 8–12 Hz · beta 13–30 Hz · 8–30 Hz |
| Ventana | 0–2 s · 0,5–2,5 s · 1–3 s · 0,5–3,5 s |
| Adaptación | sin alinear · alineación euclídea por sujeto (He y Wu, 2020) |

Los resultados se calculan con `experimentos/etapa4.py` (unas 3 horas) y quedan por sujeto en `resultados/etapa4/`. Este notebook solo los analiza.

**ITR:** se calcula por sujeto con la fórmula de Wolpaw a partir de la balanced accuracy, tomando como tiempo por decisión el final de la ventana (desde la consigna hasta que hay datos para decidir). Por eso una ventana más corta o más temprana puede tener mayor ITR aunque su precisión sea menor.

**La alineación euclídea** blanquea los ensayos de cada sujeto con su covarianza media, sin usar etiquetas. Acá se calcula con todos los ensayos del sujeto (incluidos los de prueba), lo que en un uso real equivale a tener unos minutos de EEG sin etiquetar del usuario nuevo."""),
code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

from bci.pipelines import itr_bits_por_minuto, umbral_azar

RESULTADOS = Path("..") / "resultados"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#52514e"

df = pd.read_csv(RESULTADOS / "04_resultados.csv")
df["itr"] = itr_bits_por_minuto(df["balanced_accuracy"], df["ventana_fin"])
df["sobre_azar"] = df["balanced_accuracy"] >= df["n_ensayos"].map(umbral_azar)
ORDEN = ["base", "potencia_lda", "csp_svm", "csp_rf", "banda_mu", "banda_beta",
         "ventana_0.0-2.0", "ventana_1.0-3.0", "ventana_0.5-3.5", "alineacion_ea"]
ETIQUETA = {"base": "Base: CSP+LDA, 8–30 Hz, 0,5–2,5 s", "potencia_lda": "Potencia por banda + LDA",
            "csp_svm": "CSP + SVM", "csp_rf": "CSP + Random Forest", "banda_mu": "Banda mu 8–12 Hz",
            "banda_beta": "Banda beta 13–30 Hz", "ventana_0.0-2.0": "Ventana 0–2 s",
            "ventana_1.0-3.0": "Ventana 1–3 s", "ventana_0.5-3.5": "Ventana 0,5–3,5 s",
            "alineacion_ea": "Alineación euclídea"}
print(df.groupby(["config", "esquema"]).size().unstack())
"""),
md("## Tabla resumen\n\nMedias sobre los 106 sujetos. *Pérdida* = balanced accuracy dentro − entre (negativa: el modelo entre sujetos rinde mejor)."),
code("""ancho = df.pivot_table(index=["config", "sujeto"], columns="esquema",
                       values=["balanced_accuracy", "auc", "itr", "sobre_azar"])
ancho.columns = [f"{m}_{e}" for m, e in ancho.columns]
ancho = ancho.reset_index()
ancho["perdida"] = ancho["balanced_accuracy_dentro"] - ancho["balanced_accuracy_entre"]

tabla = ancho.groupby("config").agg(
    BA_dentro=("balanced_accuracy_dentro", "mean"), BA_entre=("balanced_accuracy_entre", "mean"),
    AUC_dentro=("auc_dentro", "mean"), AUC_entre=("auc_entre", "mean"),
    perdida=("perdida", "mean"),
    ITR_dentro=("itr_dentro", "mean"), ITR_entre=("itr_entre", "mean"),
    sobre_azar_dentro=("sobre_azar_dentro", "sum"), sobre_azar_entre=("sobre_azar_entre", "sum"),
).loc[ORDEN]
tabla.index = [ETIQUETA[c] for c in tabla.index]
tabla.to_csv(RESULTADOS / "04_tabla_resumen.csv")
tabla.round(3)
"""),
md("""## Pruebas estadísticas

Para cada factor, prueba de Friedman entre todos sus niveles (incluida la base) y luego cada nivel contra la base con Wilcoxon pareado, corrigiendo por comparaciones múltiples con Holm. Todas las pruebas son pareadas por sujeto."""),
code("""def holm(p):
    p = np.asarray(p); orden = np.argsort(p); m = len(p)
    ajust = np.empty(m); acum = 0
    for rango, i in enumerate(orden):
        acum = max(acum, (m - rango) * p[i]); ajust[i] = min(1, acum)
    return ajust

FACTORES = {"modelo": ["base", "potencia_lda", "csp_svm", "csp_rf"],
            "banda": ["base", "banda_mu", "banda_beta"],
            "ventana": ["base", "ventana_0.0-2.0", "ventana_1.0-3.0", "ventana_0.5-3.5"],
            "adaptacion": ["base", "alineacion_ea"]}
filas = []
for esquema in ["dentro", "entre"]:
    for metrica in ["balanced_accuracy", "auc"]:
        col = f"{metrica}_{esquema}"
        mat = ancho.pivot(index="sujeto", columns="config", values=col)
        for factor, niveles in FACTORES.items():
            p_fr = friedmanchisquare(*[mat[n] for n in niveles]).pvalue if len(niveles) > 2 else np.nan
            ps = [wilcoxon(mat[n], mat["base"]).pvalue for n in niveles[1:]]
            for n, p, pa in zip(niveles[1:], ps, holm(ps)):
                filas.append({"esquema": esquema, "métrica": metrica, "factor": factor,
                              "nivel": ETIQUETA[n], "diferencia vs base": (mat[n] - mat["base"]).mean(),
                              "p Friedman (factor)": p_fr, "p Wilcoxon": p, "p Holm": pa})
pruebas = pd.DataFrame(filas)
pruebas.to_csv(RESULTADOS / "04_pruebas.csv", index=False)
with pd.option_context("display.float_format", "{:.3g}".format):
    display(pruebas[pruebas["métrica"] == "balanced_accuracy"])
"""),
md("""### H2: ¿CSP generaliza mejor que la potencia por banda?

Se compara CSP + LDA (base) contra potencia por banda + LDA en los dos esquemas, y también la **pérdida** de cada uno (cuánto cae al pasar a entre sujetos)."""),
code("""m = {c: ancho[ancho.config == c].set_index("sujeto") for c in ["base", "potencia_lda"]}
for col, nombre in [("balanced_accuracy_dentro", "BA dentro"), ("balanced_accuracy_entre", "BA entre"),
                    ("auc_entre", "AUC entre"), ("perdida", "pérdida")]:
    a, b = m["base"][col], m["potencia_lda"][col]
    print(f"{nombre:10s} CSP {a.mean():.3f} | potencia {b.mean():.3f} | dif {a.mean() - b.mean():+.3f} | "
          f"Wilcoxon p = {wilcoxon(a, b).pvalue:.2g}")
"""),
md("""## Figura 1 · Balanced accuracy por configuración

Media sobre los 106 sujetos con su IC 95 %. La figura la genera `experimentos/figura_etapa4.py` (es la misma que va en el informe). Los intervalos se superponen porque la variabilidad entre sujetos es grande; las comparaciones que importan son las pareadas por sujeto de la tabla anterior."""),
code("""from IPython.display import Image
from experimentos.figura_etapa4 import main as figura_configuraciones

figura_configuraciones()
Image(RESULTADOS / "04_configuraciones.png", width=800)
"""),
md("""## Figura 2 · Ventana temporal: precisión contra velocidad

Para la BCI no alcanza con acertar: también importa cuánto tarda en decidir. El ITR combina las dos cosas."""),
code("""rng = np.random.default_rng(0)
def ic95(v, n=2000):
    v = np.asarray(v); medias = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(medias, [2.5, 97.5])

vent = ["ventana_0.0-2.0", "base", "ventana_1.0-3.0", "ventana_0.5-3.5"]
nombres = ["0–2 s", "0,5–2,5 s", "1–3 s", "0,5–3,5 s"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, metrica, titulo in [(axes[0], "balanced_accuracy", "Balanced accuracy"), (axes[1], "itr", "ITR (bits/min)")]:
    for esquema, color, desplaz, et in [("dentro", NARANJA, -0.08, "Dentro"), ("entre", AZUL, 0.08, "Entre")]:
        col = f"{metrica}_{esquema}"
        medias = [ancho.loc[ancho.config == c, col].mean() for c in vent]
        ics = np.array([ic95(ancho.loc[ancho.config == c, col]) for c in vent])
        x = np.arange(len(vent)) + desplaz
        ax.errorbar(x, medias, yerr=[np.array(medias) - ics[:, 0], ics[:, 1] - np.array(medias)],
                    fmt="o", color=color, ms=7, mec="white", mew=1.2, capsize=0, lw=2, label=et)
    ax.set_xticks(range(len(vent)), nombres)
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(RESULTADOS / "04_ventanas.png", dpi=200)
"""),
md(lectura),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
