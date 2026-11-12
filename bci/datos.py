"""Carga y segmentación del dataset EEG Motor Movement/Imagery (PhysioNet).

Todos los notebooks del grupo deben cargar los datos con estas funciones,
para que el preprocesamiento sea el mismo en las tres partes del trabajo.
"""

from pathlib import Path

import mne
from mne.datasets import eegbci

# Los datos se descargan dentro del proyecto (carpeta ignorada por git).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Runs de imaginería motora: abrir/cerrar el puño izquierdo (T1) o el derecho (T2).
RUNS_IMAGINERIA = [4, 8, 12]
EVENTOS = {"izquierda": 1, "derecha": 2}
_ANOTACIONES = {"T1": EVENTOS["izquierda"], "T2": EVENTOS["derecha"]}

# Grabados a 128 Hz en vez de 160 Hz (ver resultados/sujetos.csv, generado por verificar_sujetos.py).
SUJETOS_EXCLUIDOS = (88, 92, 100)
SUJETOS = [s for s in range(1, 110) if s not in SUJETOS_EXCLUIDOS]


def cargar_raw(sujeto, runs=RUNS_IMAGINERIA):
    """Devuelve el registro continuo de un sujeto (1 a 109), con los runs concatenados."""
    DATA_DIR.mkdir(exist_ok=True)
    archivos = eegbci.load_data(sujeto, runs, path=DATA_DIR, update_path=False, verbose="error")
    raw = mne.concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True, verbose="error") for f in archivos]
    )
    eegbci.standardize(raw)  # "C3.." -> "C3"
    raw.set_montage("standard_1005")
    return raw


def epocas(raw, l_freq=None, h_freq=None, tmin=-1.0, tmax=4.0, baseline=None, referencia=None):
    """Segmenta el registro en ensayos de izquierda/derecha.

    t = 0 es el inicio de la consigna; cada ensayo de imaginería dura unos 4 s.
    Si se indican l_freq/h_freq, filtra (FIR pasa-banda) antes de segmentar.
    referencia="promedio" re-referencia al promedio común (CAR) antes de filtrar.
    """
    if referencia == "promedio":
        raw = raw.copy().set_eeg_reference("average", verbose="error")
    elif referencia is not None:
        raise ValueError(f"referencia desconocida: {referencia!r}")
    if l_freq is not None or h_freq is not None:
        raw = raw.copy().filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    eventos, _ = mne.events_from_annotations(raw, event_id=_ANOTACIONES, verbose="error")
    return mne.Epochs(
        raw, eventos, event_id=EVENTOS, tmin=tmin, tmax=tmax,
        picks="eeg", baseline=baseline, preload=True, verbose="error",
    )
