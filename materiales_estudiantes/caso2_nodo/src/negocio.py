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
    si_se_iba = (
        s["acepta_si_se_iba"] * (s["valor_cliente_retenido"] - s["costo_descuento"])
        - s["costo_contacto"]
    )
    si_se_quedaba = (
        -s["acepta_si_se_quedaba"] * s["costo_descuento"]
        - s["costo_contacto"]
    )
    return si_se_iba, si_se_quedaba


def probabilidad_de_equilibrio(s=SUPUESTOS):
    """Probabilidad de baja p a partir de la cual contactar tiene resultado esperado positivo:
    el valor de p que hace p · si_se_iba + (1 - p) · si_se_quedaba = 0.
    """
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)
    return -si_se_quedaba / (si_se_iba - si_se_quedaba)


def ganancia_campania(y_real, contactar, s=SUPUESTOS):
    """Ganancia total, en guaraníes, de contactar a los clientes marcados con True en `contactar`,
    sabiendo quiénes se fueron realmente (`y_real`, 1 = se fue). Devuelve un float.
    """
    y_real = np.asarray(y_real)
    contactar = np.asarray(contactar, dtype=bool)

    si_se_iba, si_se_quedaba = resultado_por_contacto(s)

    contactados_que_se_iban = np.sum(contactar & (y_real == 1))
    contactados_que_se_quedaban = np.sum(contactar & (y_real == 0))

    return float(
        contactados_que_se_iban * si_se_iba
        + contactados_que_se_quedaban * si_se_quedaba
    )


def mejor_umbral(y_real, prob, s=SUPUESTOS, max_contactos=None):
    """Busca el umbral u que maximiza la ganancia al contactar a todos los clientes con prob >= u.

    - Los umbrales candidatos son los valores distintos de `prob`.
    - Si max_contactos no es None, se descartan los umbrales que contactan a más clientes.
    - Devuelve un dict {"umbral": u, "ganancia": g, "contactados": n}.
    - Si ningún umbral da ganancia positiva, devuelve {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}.
    """
    y_real = np.asarray(y_real)
    prob = np.asarray(prob, dtype=float)
    si_se_iba, si_se_quedaba = resultado_por_contacto(s)

    # Clientes ordenados de mayor a menor probabilidad
    orden = np.argsort(-prob, kind="stable")
    prob_ord = prob[orden]
    se_iba_ord = (y_real[orden] == 1)

    # Para cada posición: cuántos contactados en total y cuántos de ellos se iban a ir
    contactados_acum = np.arange(1, len(prob_ord) + 1)
    se_iba_acum = np.cumsum(se_iba_ord)

    # Un umbral candidato por valor distinto de prob: se toma la última posición de cada valor,
    # así quedan contactados todos los que tienen prob >= u (incluidos los empates)
    ultimo_de_cada_valor = np.flatnonzero(np.append(prob_ord[1:] != prob_ord[:-1], True))
    umbrales = prob_ord[ultimo_de_cada_valor]
    contactados = contactados_acum[ultimo_de_cada_valor]
    iban = se_iba_acum[ultimo_de_cada_valor]
    ganancias = iban * si_se_iba + (contactados - iban) * si_se_quedaba

    # Restricción de capacidad
    if max_contactos is not None:
        permitidos = contactados <= max_contactos
        umbrales, contactados, ganancias = umbrales[permitidos], contactados[permitidos], ganancias[permitidos]

    if len(ganancias) == 0 or ganancias.max() <= 0:
        return {"umbral": np.inf, "ganancia": 0.0, "contactados": 0}

    mejor = int(np.argmax(ganancias))
    return {
        "umbral": float(umbrales[mejor]),
        "ganancia": float(ganancias[mejor]),
        "contactados": int(contactados[mejor]),
    }


def lift(y_real, prob, fraccion):
    """Tasa de positivos entre la `fraccion` de clientes con mayor prob, dividida por la tasa general.

    Ejemplo: lift(y, prob, 0.07) compara la tasa de baja del 7% de mayor riesgo con la tasa de toda la base.
    La cantidad de clientes del grupo es round(fraccion · n), con un mínimo de 1.
    """
    y_real = np.asarray(y_real)
    prob = np.asarray(prob, dtype=float)
    n = len(y_real)

    tamano_grupo = min(n, max(1, int(round(fraccion * n))))

    # Los clientes de mayor probabilidad (orden descendente)
    orden = np.argsort(-prob, kind="stable")
    tasa_grupo = np.mean(y_real[orden[:tamano_grupo]] == 1)
    tasa_general = np.mean(y_real == 1)

    return float(tasa_grupo / tasa_general)