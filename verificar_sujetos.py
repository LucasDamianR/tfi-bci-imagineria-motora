"""Descarga los 109 sujetos y resume frecuencia de muestreo y ensayos por clase.

Sirve para decidir (y documentar) qué sujetos se excluyen del análisis.
Uso:  python verificar_sujetos.py   ->  escribe resultados/sujetos.csv
"""

from pathlib import Path

import pandas as pd

from bci.datos import cargar_raw, epocas

filas = []
for sujeto in range(1, 110):
    raw = cargar_raw(sujeto)
    ep = epocas(raw)
    filas.append({
        "sujeto": sujeto,
        "sfreq": raw.info["sfreq"],
        "duracion_s": round(raw.times[-1], 1),
        "izquierda": len(ep["izquierda"]),
        "derecha": len(ep["derecha"]),
    })
    print(filas[-1], flush=True)

salida = Path(__file__).parent / "resultados"
salida.mkdir(exist_ok=True)
pd.DataFrame(filas).to_csv(salida / "sujetos.csv", index=False)
