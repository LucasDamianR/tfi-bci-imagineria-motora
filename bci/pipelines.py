"""Clasificadores y esquemas de validación comunes a todo el grupo.

- `cargar_dataset` arma X, y y el sujeto de cada ensayo para todos los sujetos,
  con el preprocesamiento de `bci.datos` (y lo guarda en caché en data/cache).
- `crear_pipeline` devuelve un clasificador de scikit-learn por nombre.
- `evaluar_intra` valida dentro de cada sujeto (validación cruzada repetida).
- `evaluar_entre` valida dejando un sujeto afuera (leave-subject-out).

Las dos evaluaciones devuelven una fila por sujeto con balanced accuracy y AUC,
para poder compararlas de a pares.
"""

import hashlib
import json

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binom
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bci.datos import DATA_DIR, EVENTOS, SUJETOS, cargar_raw, epocas

CACHE_DIR = DATA_DIR / "cache"

# CSP imprime varias líneas por ajuste; con miles de ajustes tapa la salida.
mne.set_log_level("ERROR")

# Preprocesamiento base de las etapas 2 y 3. La etapa 4 varía banda y ventana.
PREPROC_BASE = {"l_freq": 8.0, "h_freq": 30.0, "tmin": 0.5, "tmax": 2.5, "referencia": "promedio"}


def cargar_dataset(sujetos=SUJETOS, l_freq=8.0, h_freq=30.0, tmin=0.5, tmax=2.5, referencia="promedio"):
    """Devuelve X (ensayos x canales x muestras, float32), y (0 = izquierda, 1 = derecha) y grupos (sujeto).

    El filtrado se aplica al registro continuo y después se recorta la ventana [tmin, tmax] s.
    """
    params = {"sujetos": list(sujetos), "l_freq": l_freq, "h_freq": h_freq,
              "tmin": tmin, "tmax": tmax, "referencia": referencia}
    clave = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    archivo = CACHE_DIR / f"dataset_{clave}.npz"
    if archivo.exists():
        d = np.load(archivo)
        return d["X"], d["y"], d["grupos"]

    Xs, ys, gs = [], [], []
    for s in sujetos:
        ep = epocas(cargar_raw(s), l_freq=l_freq, h_freq=h_freq, tmin=tmin, tmax=tmax,
                    referencia=referencia)
        Xs.append(ep.get_data(copy=False).astype(np.float32))
        ys.append((ep.events[:, 2] == EVENTOS["derecha"]).astype(int))
        gs.append(np.full(len(ep), s))
    X, y, grupos = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(archivo, X=X, y=y, grupos=grupos)
    return X, y, grupos


# Canales sobre la corteza motora (filas FC, C y CP del sistema 10-10), en el orden de los datos.
CANALES_MOTORES = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
                   "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]


def indices_canales(nombres):
    """Posición de cada canal en X (los datos tienen los 64 canales en el orden de PhysioNet)."""
    todos = cargar_raw(1).copy().pick("eeg").ch_names
    return [todos.index(c) for c in nombres]


class PotenciaBanda(BaseEstimator, TransformerMixin):
    """Log-potencia por canal en cada banda (mu y beta por defecto), sobre un subconjunto de canales."""

    def __init__(self, canales=None, bandas=((8, 12), (13, 30)), sfreq=160.0):
        self.canales = canales
        self.bandas = bandas
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X if self.canales is None else X[:, self.canales, :]
        feats = []
        for lo, hi in self.bandas:
            sos = butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
            feats.append(np.log(np.var(sosfiltfilt(sos, Xc, axis=-1), axis=-1)))
        return np.concatenate(feats, axis=1)


def crear_pipeline(nombre="csp_lda", n_componentes=6):
    """Clasificador por nombre.

    - "csp_lda": CSP (log-varianza) + LDA. Es la configuración base.
    - "csp_svm": CSP + SVM con kernel RBF (características estandarizadas).
    - "csp_rf": CSP + Random Forest (300 árboles).
    - "potencia_lda": log-potencia en mu y beta de los 21 canales motores + LDA con shrinkage.
    """
    csp = CSP(n_components=n_componentes, reg="ledoit_wolf", log=True)
    if nombre == "csp_lda":
        return make_pipeline(csp, LinearDiscriminantAnalysis())
    if nombre == "csp_svm":
        return make_pipeline(csp, StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    if nombre == "csp_rf":
        return make_pipeline(csp, RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                         random_state=0, n_jobs=1))
    if nombre == "potencia_lda":
        return make_pipeline(PotenciaBanda(canales=indices_canales(CANALES_MOTORES)),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise ValueError(f"pipeline desconocido: {nombre!r}")


def _metricas(modelo, X, y):
    if hasattr(modelo, "decision_function"):
        puntaje = modelo.decision_function(X)
    else:
        puntaje = modelo.predict_proba(X)[:, 1]
    return balanced_accuracy_score(y, modelo.predict(X)), roc_auc_score(y, puntaje)


def alinear_euclideo(X, grupos):
    """Alineación euclídea (He y Wu, 2020): blanquea los ensayos de cada sujeto con la inversa de la
    raíz de su covarianza media, para que todos los sujetos queden centrados en la identidad.

    No usa etiquetas: solo los ensayos del propio sujeto. Con referencia promedio la covarianza
    pierde un rango, así que se usa la pseudo-inversa (se descartan autovalores casi nulos).
    """
    Xa = np.empty_like(X)
    for s in np.unique(grupos):
        m = grupos == s
        Xs = X[m].astype(np.float64)
        R = np.mean([x @ x.T / x.shape[1] for x in Xs], axis=0)
        val, vec = np.linalg.eigh(R)
        inv_raiz = np.where(val > val.max() * 1e-8, 1 / np.sqrt(np.clip(val, 1e-30, None)), 0.0)
        W = (vec * inv_raiz) @ vec.T
        Xa[m] = np.einsum("ij,njt->nit", W, Xs).astype(X.dtype)
    return Xa


def itr_bits_por_minuto(precision, segundos_por_decision, n_clases=2):
    """Tasa de transferencia de información de Wolpaw. Precisión por debajo del azar cuenta como 0."""
    p = np.clip(np.asarray(precision, dtype=float), 1e-12, 1 - 1e-12)
    bits = np.log2(n_clases) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_clases - 1))
    bits = np.where(p <= 1 / n_clases, 0.0, bits)
    return bits * 60.0 / segundos_por_decision


def _intra_un_sujeto(pipe, X, y, n_splits, n_repeats, semilla):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    res = []
    for tr, te in cv.split(X, y):
        modelo = clone(pipe).fit(X[tr], y[tr])
        res.append(_metricas(modelo, X[te], y[te]))
    return np.mean(res, axis=0)


def evaluar_intra(pipe, X, y, grupos, n_splits=5, n_repeats=10, semilla=0, n_jobs=-1):
    """Validación cruzada estratificada repetida dentro de cada sujeto. Una fila por sujeto."""
    sujetos = np.unique(grupos)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_intra_un_sujeto)(pipe, X[grupos == s], y[grupos == s], n_splits, n_repeats, semilla)
        for s in sujetos
    )
    return pd.DataFrame({
        "sujeto": sujetos,
        "n_ensayos": [int((grupos == s).sum()) for s in sujetos],
        "balanced_accuracy": [r[0] for r in res],
        "auc": [r[1] for r in res],
    })


def _entre_un_pliegue(pipe, X, y, tr, te):
    modelo = clone(pipe).fit(X[tr], y[tr])
    return _metricas(modelo, X[te], y[te])


def evaluar_entre(pipe, X, y, grupos, n_jobs=4):
    """Leave-subject-out: entrena con todos los sujetos menos uno y evalúa en ese. Una fila por sujeto.

    Cada pliegue copia el entrenamiento (~105 sujetos) y CSP lo pasa a float64: unos 2 GB por
    proceso. Por eso n_jobs es 4 por defecto y no -1.
    """
    logo = LeaveOneGroupOut()
    pliegues = list(logo.split(X, y, grupos))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_entre_un_pliegue)(pipe, X, y, tr, te) for tr, te in pliegues
    )
    sujetos = [int(grupos[te][0]) for _, te in pliegues]
    return pd.DataFrame({
        "sujeto": sujetos,
        "n_ensayos": [len(te) for _, te in pliegues],
        "balanced_accuracy": [r[0] for r in res],
        "auc": [r[1] for r in res],
    })


def umbral_azar(n_ensayos, alfa=0.05):
    """Accuracy mínima para superar el azar (p < alfa, binomial unilateral, 2 clases balanceadas)."""
    # Menor k con P(X >= k) < alfa, es decir, ppf(1 - alfa) + 1 aciertos.
    return (binom.ppf(1 - alfa, n_ensayos, 0.5) + 1) / n_ensayos
