"""Evaluación económica de la campaña de retención de Nodo Internet.

Completá las funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_negocio.py
"""
import numpy as np

# Supuestos acordados por Finanzas y Fidelización (ver el enunciado del caso)
SUPUESTOS = {
    "costo_contacto": 10_000,            # Gs. por cliente contactado
    "costo_descuento": 200_000,          # Gs. por cliente que acepta el descuento
    "valor_cliente_retenido": 1_000_000,  # Gs. de margen de un cliente que se iba a ir y se queda
    "acepta_si_se_iba": 0.40,            # proporción que acepta la oferta entre los que se iban a ir
    "acepta_si_se_quedaba": 0.60,        # proporción que acepta la oferta entre los que se iban a quedar
}


def resultado_por_contacto(s=SUPUESTOS):
    """Devuelve una tupla (si_se_iba, si_se_quedaba) con el resultado esperado, en guaraníes, de contactar a:
    - un cliente que se iba a ir (si acepta, se queda y deja su margen, menos el descuento), y
    - un cliente que se iba a quedar (si acepta, la empresa resigna el descuento).
    En los dos casos se paga el costo del contacto. No contactar tiene resultado 0.
    """
    raise NotImplementedError("Completar resultado_por_contacto")


def probabilidad_de_equilibrio(s=SUPUESTOS):
    """Probabilidad de baja p a partir de la cual contactar tiene resultado esperado positivo:
    el valor de p que hace p · si_se_iba + (1 - p) · si_se_quedaba = 0.
    """
    raise NotImplementedError("Completar probabilidad_de_equilibrio")


def ganancia_campania(y_real, contactar, s=SUPUESTOS):
    """Ganancia total, en guaraníes, de contactar a los clientes marcados con True en `contactar`,
    sabiendo quiénes se fueron realmente (`y_real`, 1 = se fue). Devuelve un float.
    """
    raise NotImplementedError("Completar ganancia_campania")


def mejor_umbral(y_real, prob, s=SUPUESTOS, max_contactos=None):
    """Busca el umbral u que maximiza la ganancia al contactar a todos los clientes con prob >= u.

    - Los umbrales candidatos son los valores distintos de `prob`.
    - Si max_contactos no es None, se descartan los umbrales que contactan a más clientes.
    - Devuelve un dict {"umbral": u, "ganancia": g, "contactados": n}.
    - Si ningún umbral da ganancia positiva, devuelve {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}.
    """
    raise NotImplementedError("Completar mejor_umbral")


def lift(y_real, prob, fraccion):
    """Tasa de positivos entre la `fraccion` de clientes con mayor prob, dividida por la tasa general.

    Ejemplo: lift(y, prob, 0.07) compara la tasa de baja del 7% de mayor riesgo con la tasa de toda la base.
    La cantidad de clientes del grupo es round(fraccion · n), con un mínimo de 1.
    """
    raise NotImplementedError("Completar lift")
