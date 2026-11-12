"""Etapa 5: curva de calibración. ¿Cuántos ensayos del sujeto nuevo hacen falta, y para qué?

Para cada sujeto que se deja afuera se toman k ensayos de calibración (estratificados por clase)
y se evalúa con el resto de sus ensayos. Estrategias:

- general:          modelo entrenado con los otros 105 sujetos, sin usar nada del sujeto nuevo.
- general_umbral:   el mismo modelo, pero con el umbral de decisión recentrado con los k ensayos
                    (punto medio entre los puntajes medios de cada clase). No cambia el AUC.
- alineado_umbral:  modelo entrenado con sujetos alineados (alineación euclídea); el sujeto nuevo
                    se alinea con la covarianza de sus k ensayos de calibración, y se recentra el umbral.
- personal:         CSP + LDA entrenado solo con los k ensayos del sujeto (k >= 10).

Se repite 10 veces por k con distintos ensayos de calibración. Configuración base del notebook 02.
Resultados por sujeto, k y repetición en resultados/05_calibracion.csv.

Uso (desde la raíz del repo):  python -m experimentos.etapa5
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from bci.pipelines import PREPROC_BASE, alinear_euclideo, cargar_dataset, crear_pipeline

SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "05_calibracion.csv"
KS = [0, 5, 10, 20, 30]
REPETICIONES = 10


def _blanqueador(Xcal):
    R = np.mean([x @ x.T / x.shape[1] for x in Xcal.astype(np.float64)], axis=0)
    val, vec = np.linalg.eigh(R)
    inv_raiz = np.where(val > val.max() * 1e-8, 1 / np.sqrt(np.clip(val, 1e-30, None)), 0.0)
    return (vec * inv_raiz) @ vec.T


def _umbral(puntaje_cal, y_cal):
    """Desplazamiento que deja el corte en el punto medio de los puntajes medios de cada clase."""
    return -(puntaje_cal[y_cal == 0].mean() + puntaje_cal[y_cal == 1].mean()) / 2


def _metricas(y, puntaje):
    return balanced_accuracy_score(y, (puntaje > 0).astype(int)), roc_auc_score(y, puntaje)


def _un_sujeto(s, X, Xa, y, grupos):
    tr, te = grupos != s, grupos == s
    pipe = crear_pipeline("csp_lda")
    general = clone(pipe).fit(X[tr], y[tr])
    alineado = clone(pipe).fit(Xa[tr], y[tr])
    Xs, ys = X[te], y[te]
    filas = []
    for k in KS:
        if k == 0:
            divisiones = [(np.array([], int), np.arange(len(ys)))]
        else:
            sss = StratifiedShuffleSplit(n_splits=REPETICIONES, train_size=k, random_state=s)
            divisiones = list(sss.split(Xs, ys))
        for rep, (cal, prueba) in enumerate(divisiones):
            p_gen = general.decision_function(Xs)
            res = {"general": _metricas(ys[prueba], p_gen[prueba])}
            if k > 0:
                b = _umbral(p_gen[cal], ys[cal])
                res["general_umbral"] = _metricas(ys[prueba], p_gen[prueba] + b)
                W = _blanqueador(Xs[cal])
                Xs_al = np.einsum("ij,njt->nit", W, Xs.astype(np.float64))
                p_al = alineado.decision_function(Xs_al)
                res["alineado_umbral"] = _metricas(ys[prueba], p_al[prueba] + _umbral(p_al[cal], ys[cal]))
            if k >= 10:
                personal = clone(pipe).fit(Xs[cal], ys[cal])
                res["personal"] = _metricas(ys[prueba], personal.decision_function(Xs[prueba]))
            for estrategia, (ba, auc) in res.items():
                filas.append({"sujeto": int(s), "k": k, "rep": rep, "estrategia": estrategia,
                              "n_prueba": len(prueba), "balanced_accuracy": ba, "auc": auc})
    return filas


def main():
    t0 = time.time()
    X, y, grupos = cargar_dataset(**PREPROC_BASE)
    Xa = alinear_euclideo(X, grupos)  # para entrenar: cada sujeto de entrenamiento con todos sus ensayos
    print(f"datos {X.shape} en {time.time() - t0:.0f} s", flush=True)
    # Dos modelos generales por sujeto (~4 GB por proceso): 3 procesos en paralelo.
    res = Parallel(n_jobs=3, verbose=5)(
        delayed(_un_sujeto)(s, X, Xa, y, grupos) for s in np.unique(grupos)
    )
    df = pd.DataFrame([f for filas in res for f in filas])
    df.to_csv(SALIDA, index=False)
    resumen = df.groupby(["k", "estrategia"])[["balanced_accuracy", "auc"]].mean().round(3)
    print(resumen.to_string())
    print(f"listo en {(time.time() - t0) / 60:.1f} min:", SALIDA)


if __name__ == "__main__":
    main()
