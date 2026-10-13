"""Regla de derivación de mensajes e indicadores de negocio."""
import numpy as np

REVISION = "revision_humana"


def derivar(proba, clases, umbral_fraude, umbral_confianza):
    proba = np.asarray(proba, dtype=float)
    clases = np.asarray(clases)
    if proba.ndim != 2 or proba.shape[1] != len(clases):
        raise ValueError("Las columnas de proba deben corresponder a clases")
    destinos = np.full(proba.shape[0], REVISION, dtype=object)
    idx_fraude = np.flatnonzero(clases == "fraude")
    idx_max = np.argmax(proba, axis=1)
    maximos = proba[np.arange(len(proba)), idx_max]
    if idx_fraude.size:
        alerta = proba[:, idx_fraude[0]] >= umbral_fraude
        destinos[alerta] = "fraude"
    else:
        alerta = np.zeros(len(proba), dtype=bool)
    seguros = (~alerta) & (maximos >= umbral_confianza)
    destinos[seguros] = clases[idx_max[seguros]]
    return destinos.astype(str)


def evaluar_derivacion(y_real, destino):
    y_real = np.asarray(y_real, dtype=str)
    destino = np.asarray(destino, dtype=str)
    if len(y_real) != len(destino):
        raise ValueError("y_real y destino deben tener la misma longitud")
    automaticos = destino != REVISION
    cobertura = float(np.mean(automaticos)) if len(destino) else 0.0
    exactitud = float(np.mean(destino[automaticos] == y_real[automaticos])) if automaticos.any() else 0.0
    fraudes = y_real == "fraude"
    fraude_perdido = float(np.mean(automaticos[fraudes] & (destino[fraudes] != "fraude"))) if fraudes.any() else 0.0
    return {"cobertura": cobertura, "exactitud_automatica": exactitud, "fraude_perdido": fraude_perdido}
