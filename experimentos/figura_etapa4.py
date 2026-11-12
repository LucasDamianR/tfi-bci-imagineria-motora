"""Figura de la etapa 4: balanced accuracy media por configuración, dentro del sujeto y entre sujetos.

Uso (desde la raíz del repo):
    python -m experimentos.figura_etapa4                 # informe: resultados/04_configuraciones.png
    python -m experimentos.figura_etapa4 --presentacion  # etiquetas simples: resultados/presentacion/04_configuraciones.png
Lee resultados/04_resultados.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "resultados"
TINTA, TINTA_2, GRILLA, FONDO = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"
AZUL, NARANJA = "#2a78d6", "#eb6834"

TEXTOS = {
    "informe": {
        "etiquetas": {
            "base": "Base: CSP + LDA, 8–30 Hz, 0,5–2,5 s",
            "alineacion_ea": "Alineación euclídea",
            "csp_svm": "Modelo: CSP + SVM",
            "csp_rf": "Modelo: CSP + Random Forest",
            "potencia_lda": "Modelo: potencia mu/beta + LDA (sin CSP)",
            "banda_mu": "Banda: solo mu (8–12 Hz)",
            "banda_beta": "Banda: solo beta (13–30 Hz)",
            "ventana_0.0-2.0": "Ventana: 0–2 s",
            "ventana_0.5-3.5": "Ventana: 0,5–3,5 s",
            "ventana_1.0-3.0": "Ventana: 1–3 s",
        },
        "entre": "Entre sujetos (leave-subject-out)",
        "dentro": "Dentro del sujeto",
        "xlabel": "Balanced accuracy media de 106 sujetos (barras: IC 95 %)",
        "titulo": "Solo la alineación euclídea mejora al modelo entre sujetos",
        "subtitulo": "Línea punteada: configuración base de cada esquema. Ordenado por el resultado entre sujetos.",
        "destacadas": ("Base", "Alineación"),
    },
    "presentacion": {
        "etiquetas": {
            "base": "Pipeline base (CSP + LDA)",
            "alineacion_ea": "Alineación euclídea",
            "csp_svm": "Otro clasificador: SVM",
            "csp_rf": "Otro clasificador: Random Forest",
            "potencia_lda": "Sin CSP: electrodos fijos",
            "banda_mu": "Solo ritmo mu (8–12 Hz)",
            "banda_beta": "Solo ritmo beta (13–30 Hz)",
            "ventana_0.0-2.0": "Analizar de 0 a 2 s",
            "ventana_0.5-3.5": "Analizar de 0,5 a 3,5 s",
            "ventana_1.0-3.0": "Analizar de 1 a 3 s",
        },
        "entre": "Modelo general",
        "dentro": "Modelo personal",
        "xlabel": "Aciertos promedio de 106 personas (0,5 = azar; barras: margen de error)",
        "titulo": "Solo la alineación euclídea mejora al modelo general",
        "subtitulo": "General: entrenado con otras personas. Personal: con la misma persona. Línea punteada: el pipeline base.",
        "destacadas": ("Pipeline base", "Alineación"),
    },
}


def main(modo="informe"):
    t = TEXTOS[modo]
    salida = RES / "04_configuraciones.png" if modo == "informe" else RES / "presentacion" / "04_configuraciones.png"
    salida.parent.mkdir(exist_ok=True)
    esquemas = {  # esquema -> (etiqueta, color, marcador, desplazamiento vertical)
        "entre": (t["entre"], AZUL, "o", -0.14),
        "dentro": (t["dentro"], NARANJA, "s", 0.14),
    }

    d = pd.read_csv(RES / "04_resultados.csv")
    g = d.groupby(["config", "esquema"]).balanced_accuracy
    media = g.mean().unstack()
    ic = (1.96 * g.std() / np.sqrt(g.count())).unstack()  # IC 95 % de la media (aprox. normal)
    orden = media["entre"].sort_values().index  # de peor a mejor: el mejor queda arriba

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    y = np.arange(len(orden))
    base = media.loc["base"]
    for esq, (nombre, color, marcador, dy) in esquemas.items():
        ax.axvline(base[esq], color=color, lw=1, ls=":", alpha=0.7, zorder=1)
        ax.errorbar(media.loc[orden, esq], y + dy, xerr=ic.loc[orden, esq], fmt=marcador,
                    color=color, ms=7, lw=1.5, capsize=0, mec=FONDO, mew=1.5, label=nombre, zorder=3)
    ax.set_yticks(y, [t["etiquetas"][c] for c in orden], color=TINTA, fontsize=9.5)
    for et in ax.get_yticklabels():
        if et.get_text().startswith(t["destacadas"]):
            et.set_fontweight("bold")
    ax.set_xlabel(t["xlabel"], color=TINTA_2, fontsize=9.5)
    ax.set_xlim(0.54, 0.72)
    ax.grid(axis="x", color=GRILLA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRILLA)
    ax.tick_params(colors=TINTA_2, length=0)
    ax.set_title(t["titulo"], loc="left", color=TINTA, fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.02, t["subtitulo"], transform=ax.transAxes, color=TINTA_2, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    print("listo:", salida)


if __name__ == "__main__":
    main("presentacion" if "--presentacion" in sys.argv else "informe")
