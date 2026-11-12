"""Figura del ERD en mu (notebook 01, sección 3) para el informe.

Curso temporal del cambio de potencia mu (8-12 Hz) respecto de la línea de base (-1 a 0 s),
en C3 y C4, por clase, promediado en los primeros 20 sujetos (la muestra exploratoria del notebook 01).

Uso (desde la raíz del repo):  python -m experimentos.figura_erd
Escribe resultados/01_erd_mu.png (tarda alrededor de un minuto).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from bci.datos import cargar_raw, epocas

RES = Path(__file__).resolve().parents[1] / "resultados"
SUJETOS = range(1, 21)
CANALES = ["C3", "C4"]
CLASES = {"izquierda": ("Imagina mano izquierda", "#eb6834"), "derecha": ("Imagina mano derecha", "#2a78d6")}
TINTA, TINTA_2, GRILLA, FONDO = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"


def curva_erd(raw, banda=(8, 12)):
    """ERD % por clase y canal: (P(t) - R) / R * 100, con R = potencia media entre -1 y 0 s."""
    ep = epocas(raw, *banda, tmin=-1.0, tmax=4.0)
    ep.apply_hilbert(envelope=True)
    potencia = ep.get_data(picks=CANALES) ** 2
    base = ep.times < 0
    out = {}
    for clase, codigo in ep.event_id.items():
        p = potencia[ep.events[:, 2] == codigo].mean(axis=0)
        r = p[:, base].mean(axis=1, keepdims=True)
        out[clase] = (p - r) / r * 100
    return ep.times, out


def main():
    mne.set_log_level("error")
    curvas = [curva_erd(cargar_raw(s)) for s in SUJETOS]
    t = curvas[0][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=FONDO)
    for i, (ax, canal) in enumerate(zip(axes, CANALES)):
        ax.set_facecolor(FONDO)
        ax.axvspan(0.5, 1.5, color=GRILLA, alpha=0.6, lw=0, zorder=0)
        ax.axhline(0, color=TINTA_2, lw=0.8)
        ax.axvline(0, color=TINTA_2, lw=0.8, ls=":")
        for clase, (nombre, color) in CLASES.items():
            media = np.mean([c[1][clase][i] for c in curvas], axis=0)
            suave = np.convolve(media, np.ones(32) / 32, mode="same")  # media móvil de 0,2 s, como el notebook 01
            print(f"{canal} {clase}: mínimo entre 0 y 2 s = {suave[(t > 0) & (t < 2)].min():.0f} %")
            ax.plot(t, suave, color=color, lw=2, label=nombre)
        ax.set_xlim(-0.85, 3.85)  # los bordes de la media móvil no son confiables
        hemis = "izquierdo" if canal == "C3" else "derecho"
        ax.set_title(f"Electrodo {canal} (hemisferio {hemis})", loc="left", color=TINTA, fontsize=11)
        ax.set_xlabel("Segundos desde que aparece la consigna en pantalla", color=TINTA_2, fontsize=9.5)
        ax.text(0.05, ax.get_ylim()[0] + 2, "consigna", color=TINTA_2, fontsize=8.5)
        ax.grid(axis="y", color=GRILLA, lw=0.8)
        for lado in ["top", "right", "left"]:
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(GRILLA)
        ax.tick_params(colors=TINTA_2, length=0)
    axes[0].set_ylabel("Potencia mu respecto del reposo (%)", color=TINTA_2, fontsize=9.5)
    axes[0].text(1.0, axes[0].get_ylim()[1] * 0.92, "0,5–1,5 s", ha="center", color=TINTA_2, fontsize=8.5)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TINTA)
    fig.suptitle("La potencia mu cae en los dos hemisferios, algo más del lado opuesto a la mano",
                 x=0.01, ha="left", color=TINTA, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.9, "0 % = potencia en reposo (el segundo antes de la consigna). Por debajo de 0, la potencia cae: es el ERD. "
             "Media de los sujetos 1 a 20.", color=TINTA_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RES / "01_erd_mu.png", dpi=200, facecolor=FONDO)
    print("listo:", RES / "01_erd_mu.png")


if __name__ == "__main__":
    main()
