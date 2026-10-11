"""Pruebas de src/negocio.py. Ejecutar desde la carpeta caso2_nodo con:  python -m pytest -q

Las pruebas usan supuestos distintos de los del caso, para verificar las fórmulas sin revelar los resultados.
"""
import numpy as np
import pytest

from src.negocio import ganancia_campania, lift, mejor_umbral, probabilidad_de_equilibrio, resultado_por_contacto

S = {
    "costo_contacto": 5_000,
    "costo_descuento": 100_000,
    "valor_cliente_retenido": 500_000,
    "acepta_si_se_iba": 0.5,
    "acepta_si_se_quedaba": 0.5,
}


def test_resultado_por_contacto():
    si_se_iba, si_se_quedaba = resultado_por_contacto(S)
    assert si_se_iba == pytest.approx(0.5 * (500_000 - 100_000) - 5_000)   # 195.000
    assert si_se_quedaba == pytest.approx(-(0.5 * 100_000) - 5_000)        # -55.000


def test_probabilidad_de_equilibrio():
    p = probabilidad_de_equilibrio(S)
    assert p == pytest.approx(55_000 / (195_000 + 55_000))
    si_se_iba, si_se_quedaba = resultado_por_contacto(S)
    assert p * si_se_iba + (1 - p) * si_se_quedaba == pytest.approx(0, abs=1e-6)


def test_ganancia_campania():
    y = np.array([1, 1, 0, 0, 0, 1])
    contactar = np.array([True, False, True, True, False, True])
    # 2 contactados que se iban a ir y 2 que se iban a quedar
    assert ganancia_campania(y, contactar, S) == pytest.approx(2 * 195_000 - 2 * 55_000)
    assert ganancia_campania(y, np.zeros(6, dtype=bool), S) == 0


def test_mejor_umbral_sin_limite():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05])
    r = mejor_umbral(y, prob, S)
    # Contactar a los 4 primeros: 3 × 195.000 - 1 × 55.000 = 530.000 (es el máximo)
    assert r["umbral"] == pytest.approx(0.6)
    assert r["ganancia"] == pytest.approx(530_000)
    assert r["contactados"] == 4


def test_mejor_umbral_con_capacidad():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05])
    r = mejor_umbral(y, prob, S, max_contactos=2)
    assert r["contactados"] == 2
    assert r["ganancia"] == pytest.approx(2 * 195_000)


def test_mejor_umbral_sin_ganancia_no_contacta():
    y = np.array([0, 0, 0, 0, 1])
    prob = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    r = mejor_umbral(y, prob, S)
    assert r["contactados"] == 0 and r["ganancia"] == 0 and r["umbral"] == np.inf


def test_lift():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    prob = np.linspace(1, 0, 10)
    assert lift(y, prob, 0.2) == pytest.approx((1 / 2) / (2 / 10))
    assert lift(y, prob, 1.0) == pytest.approx(1.0)
