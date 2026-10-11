"""Pruebas de src/derivacion.py. Ejecutar desde la carpeta caso1_cauce con:  python -m pytest -q"""
import numpy as np
import pytest

from src.derivacion import REVISION, derivar, evaluar_derivacion

CLASES = np.array(["cuenta_acceso", "fraude", "tarjeta"])
PROBA = np.array([
    [0.97, 0.02, 0.01],   # confiable -> cuenta_acceso
    [0.60, 0.30, 0.10],   # P(fraude) alta -> fraude aunque no sea la más probable
    [0.50, 0.05, 0.45],   # dudoso -> revisión humana
    [0.01, 0.98, 0.01],   # fraude
    [0.02, 0.03, 0.95],   # tarjeta
])


def test_derivar():
    destino = derivar(PROBA, CLASES, umbral_fraude=0.25, umbral_confianza=0.9)
    assert list(destino) == ["cuenta_acceso", "fraude", REVISION, "fraude", "tarjeta"]


def test_derivar_umbral_de_fraude_tiene_prioridad():
    destino = derivar(PROBA, CLASES, umbral_fraude=0.02, umbral_confianza=0.9)
    assert list(destino) == ["fraude", "fraude", "fraude", "fraude", "fraude"]


def test_derivar_sin_umbral_de_confianza_deriva_todo():
    destino = derivar(PROBA, list(CLASES), umbral_fraude=1.01, umbral_confianza=0.0)
    assert list(destino) == ["cuenta_acceso", "cuenta_acceso", "cuenta_acceso", "fraude", "tarjeta"]


def test_evaluar_derivacion():
    y_real = ["cuenta_acceso", "tarjeta", "fraude", "fraude", "fraude", "tarjeta"]
    destino = ["cuenta_acceso", "fraude", REVISION, "fraude", "tarjeta", REVISION]
    r = evaluar_derivacion(y_real, destino)
    assert r["cobertura"] == pytest.approx(4 / 6)
    assert r["exactitud_automatica"] == pytest.approx(2 / 4)
    assert r["fraude_perdido"] == pytest.approx(1 / 3)
