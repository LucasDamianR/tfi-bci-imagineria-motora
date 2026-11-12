"""Figura de la etapa 5: curva de calibración (balanced accuracy según los ensayos del sujeto nuevo).

Uso (desde la raíz del repo):
    python -m experimentos.figura_etapa5                 # informe: resultados/05_calibracion.png
    python -m experimentos.figura_etapa5 --presentacion  # etiquetas simples: resultados/presentacion/05_calibracion.png
Lee resultados/05_calibracion.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "resultados"
TINTA, TINTA_2, GRILLA, FONDO = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"
BA_DENTRO_02 = 0.604  # modelo personal con validación cruzada (~36 ensayos de entrenamiento), notebook 02

# estrategia -> (color, marcador); el orden fija el color de cada una
ESTILO = {
    "alineado_umbral": ("#2a78d6", "o"),
    "general_umbral": ("#eb6834", "s"),
    "general": ("#1baf7a", "D"),
    "personal": ("#eda100", "^"),
}
TEXTOS = {
    "informe": {
        "estrategias": {"alineado_umbral": "General alineado + umbral", "general_umbral": "General + umbral",
                        "general": "General, sin ajustar", "personal": "Personal (solo sus k ensayos)"},
        "referencia": "Personal con ~36 ensayos\n(validación cruzada, 6.2)",
        "xlabel": "Ensayos de calibración del sujeto nuevo (k)",
        "ylabel": "Balanced accuracy media de 106 sujetos",
        "titulo": "Con 5 a 10 ensayos, el modelo general alineado supera a cualquier modelo personal",
        "subtitulo": "Cada punto promedia 10 sorteos de los ensayos de calibración. Comparaciones pareadas por sujeto en el texto.",
        "margen": 0.74,
        "referencia_izquierda": False,
    },
    "presentacion": {
        "estrategias": {"alineado_umbral": "General, alineado y\ncon umbral corregido",
                        "general_umbral": "General, con umbral\ncorregido",
                        "general": "General, sin ajustar", "personal": "Propio, entrenado solo\ncon esos ensayos"},
        "referencia": "Propio con ~36 ensayos",
        "xlabel": "Ensayos de calibración que graba la persona nueva",
        "ylabel": "Aciertos promedio de 106 personas",
        "titulo": "Con 5 a 10 ensayos, el modelo general ajustado supera a cualquier modelo propio",
        "subtitulo": "Cada punto promedia 10 sorteos distintos de los ensayos de calibración.",
        "margen": 0.72,
        "referencia_izquierda": True,
    },
}


def main(modo="informe"):
    t = TEXTOS[modo]
    salida = RES / "05_calibracion.png" if modo == "informe" else RES / "presentacion" / "05_calibracion.png"
    salida.parent.mkdir(exist_ok=True)

    d = pd.read_csv(RES / "05_calibracion.csv")
    # Primero se promedian las repeticiones de cada sujeto, para que todos pesen igual.
    s = d.groupby(["sujeto", "k", "estrategia"]).balanced_accuracy.mean().reset_index()
    media = s.groupby(["estrategia", "k"]).balanced_accuracy.mean()

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    ax.axhline(0.5, color=TINTA_2, lw=1, ls="-", alpha=0.5, zorder=1)
    ax.text(30.8, 0.5, "Azar", va="center", color=TINTA_2, fontsize=9)
    ax.axhline(BA_DENTRO_02, color=TINTA_2, lw=1, ls=":", zorder=1)
    if t["referencia_izquierda"]:  # encima de la línea, a la izquierda, lejos de las etiquetas de la derecha
        ax.text(-0.6, BA_DENTRO_02 + 0.002, t["referencia"], va="bottom", color=TINTA_2, fontsize=8.5)
    else:
        ax.text(30.8, BA_DENTRO_02, t["referencia"], va="center", color=TINTA_2, fontsize=8.5)
    for est, (color, marcador) in ESTILO.items():
        m = media.loc[est]
        k = m.index.values
        if est not in ("general", "personal"):  # k = 0 es el punto de partida común de las estrategias generales
            k = np.r_[0, k]
            m = pd.concat([media.loc["general"].loc[[0]], m])
        ax.plot(k, m.values, color=color, lw=2, marker=marcador, ms=7, mec=FONDO, mew=1.5, zorder=3)
        ax.text(30.8, m.values[-1], t["estrategias"][est], va="center", color=TINTA, fontsize=9.5)
    ax.set_xticks([0, 5, 10, 20, 30])
    ax.set_xlim(-1, 30.5)
    ax.set_ylim(0.48, 0.72)
    ax.set_xlabel(t["xlabel"], color=TINTA_2, fontsize=9.5)
    ax.set_ylabel(t["ylabel"], color=TINTA_2, fontsize=9.5)
    ax.grid(axis="y", color=GRILLA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRILLA)
    ax.tick_params(colors=TINTA_2, length=0)
    ax.set_title(t["titulo"], loc="left", color=TINTA, fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.02, t["subtitulo"], transform=ax.transAxes, color=TINTA_2, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=t["margen"])
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    print("listo:", salida)


if __name__ == "__main__":
    main("presentacion" if "--presentacion" in sys.argv else "informe")
