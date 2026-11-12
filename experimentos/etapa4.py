"""Etapa 4: compara configuraciones del pipeline dentro del sujeto y entre sujetos.

Cada configuración cambia UN factor respecto de la base (CSP + LDA, 8-30 Hz, 0,5-2,5 s, sin alinear):
modelo, banda, ventana o alineación euclídea. Los resultados por sujeto quedan en
resultados/etapa4/<config>.csv; si un archivo ya existe, esa configuración no se vuelve a correr.

Uso (desde la raíz del repo):  python -m experimentos.etapa4
Tarda unas 3 horas con 4 procesos para leave-subject-out (unos 19 minutos por configuración con CSP).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bci.pipelines import alinear_euclideo, cargar_dataset, crear_pipeline, evaluar_entre, evaluar_intra

SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "etapa4"
SFREQ = 160.0

BASE = {"pipeline": "csp_lda", "banda": (8.0, 30.0), "ventana": (0.5, 2.5), "alinear": False}

# nombre -> (factor, cambios respecto de BASE)
CONFIGS = {
    "base":            ("base",       {}),
    "potencia_lda":    ("modelo",     {"pipeline": "potencia_lda"}),
    "csp_svm":         ("modelo",     {"pipeline": "csp_svm"}),
    "csp_rf":          ("modelo",     {"pipeline": "csp_rf"}),
    "banda_mu":        ("banda",      {"banda": (8.0, 12.0)}),
    "banda_beta":      ("banda",      {"banda": (13.0, 30.0)}),
    "ventana_0.0-2.0": ("ventana",    {"ventana": (0.0, 2.0)}),
    "ventana_1.0-3.0": ("ventana",    {"ventana": (1.0, 3.0)}),
    "ventana_0.5-3.5": ("ventana",    {"ventana": (0.5, 3.5)}),
    "alineacion_ea":   ("adaptacion", {"alinear": True}),
}


def recortar(X, ventana, tmin_datos=0.0):
    a = int(round((ventana[0] - tmin_datos) * SFREQ))
    b = int(round((ventana[1] - tmin_datos) * SFREQ)) + 1
    return np.ascontiguousarray(X[:, :, a:b])


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    configs = {n: (f, {**BASE, **c}) for n, (f, c) in CONFIGS.items()}
    # Agrupar por banda: cada banda se carga una sola vez, con la ventana completa 0-3,5 s (con 4 s se pierden 20 ensayos que pasan el final del registro).
    for banda in sorted({c["banda"] for _, c in configs.values()}):
        pendientes = [n for n, (_, c) in configs.items()
                      if c["banda"] == banda and not (SALIDA / f"{n}.csv").exists()]
        if not pendientes:
            continue
        t0 = time.time()
        X0, y, grupos = cargar_dataset(l_freq=banda[0], h_freq=banda[1], tmin=0.0, tmax=3.5,
                                       referencia="promedio")
        print(f"[banda {banda}] datos {X0.shape} en {time.time() - t0:.0f} s", flush=True)
        for nombre in pendientes:
            factor, c = configs[nombre]
            t0 = time.time()
            X = recortar(X0, c["ventana"])
            if c["alinear"]:
                X = alinear_euclideo(X, grupos)
            pipe = crear_pipeline(c["pipeline"])
            intra = evaluar_intra(pipe, X, y, grupos).assign(esquema="dentro")
            entre = evaluar_entre(pipe, X, y, grupos).assign(esquema="entre")
            res = pd.concat([intra, entre]).assign(
                config=nombre, factor=factor, pipeline=c["pipeline"],
                banda=f"{c['banda'][0]:g}-{c['banda'][1]:g}",
                ventana_ini=c["ventana"][0], ventana_fin=c["ventana"][1], alineacion=c["alinear"],
            )
            res.to_csv(SALIDA / f"{nombre}.csv", index=False)
            print(f"  {nombre:16s} dentro BA {intra.balanced_accuracy.mean():.3f} | "
                  f"entre BA {entre.balanced_accuracy.mean():.3f} AUC {entre.auc.mean():.3f} | "
                  f"{time.time() - t0:.0f} s", flush=True)
            del X
        del X0

    todo = pd.concat([pd.read_csv(SALIDA / f"{n}.csv") for n in CONFIGS])
    todo.to_csv(SALIDA.parent / "04_resultados.csv", index=False)
    print("listo:", SALIDA.parent / "04_resultados.csv")


if __name__ == "__main__":
    main()
