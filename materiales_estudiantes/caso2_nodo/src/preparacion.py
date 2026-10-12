"""Preparación de los datos de clientes de Nodo Internet.

Completá las dos funciones con pandas y la biblioteca estándar.
Verificá tu trabajo con:  python -m pytest -q tests/test_preparacion.py
"""
import unicodedata

import numpy as np
import pandas as pd

METODOS_VALIDOS = ["debito_automatico", "tarjeta_credito", "transferencia", "efectivo"]
DIAS_POR_MES = 30.44


def normalizar_metodo_pago(valor):
    """Lleva las variantes con que el CRM guarda el medio de pago a uno de METODOS_VALIDOS.

    Ejemplos: "Débito automático", "DEBITO AUTOMATICO" y "Debito_Automatico" -> "debito_automatico";
    "Tarjeta de crédito" y "tarjeta credito" -> "tarjeta_credito"; "transferencia bancaria" -> "transferencia";
    "efectivo " -> "efectivo".
    Si el valor no corresponde a ninguno de los cuatro, lanzar ValueError.
    """
    if not isinstance(valor, str):
        raise ValueError(f"Método de pago inválido: {valor!r}")

    # Quitar tildes (descomposición Unicode y descarte de marcas diacríticas)
    texto = unicodedata.normalize("NFKD", valor)
    texto = "".join(c for c in texto if not unicodedata.combining(c))

    # Minúsculas, separadores unificados y espacios colapsados
    texto = texto.lower().replace("_", " ").replace("-", " ")
    texto = " ".join(texto.split())

    if texto.startswith("debito"):
        return "debito_automatico"
    if texto.startswith("tarjeta"):
        return "tarjeta_credito"
    if texto.startswith("transferencia"):
        return "transferencia"
    if texto.startswith("efectivo"):
        return "efectivo"

    raise ValueError(f"Método de pago inválido: {valor!r}")


def agregar_variables(df):
    """Devuelve una COPIA de df (sin modificar el original) con metodo_pago normalizado y tres columnas nuevas.

    Todas se calculan a la fecha_corte de cada fila, nunca a la fecha de hoy:
    - antiguedad_meses: días entre fecha_alta y fecha_corte, divididos por DIAS_POR_MES.
    - meses_hasta_fin_permanencia: días entre fecha_corte y fin_permanencia, divididos por DIAS_POR_MES.
      Vale 0 en los contratos mensuales (no tienen fecha de fin de permanencia).
    - sin_permanencia_3m: 1 si el cliente puede irse sin penalidad en los próximos 3 meses, es decir,
      si su contrato es mensual o si meses_hasta_fin_permanencia <= 3; 0 en otro caso (entero).
    """
    resultado = df.copy()

    resultado["metodo_pago"] = resultado["metodo_pago"].map(normalizar_metodo_pago)

    fecha_corte = pd.to_datetime(resultado["fecha_corte"])
    fecha_alta = pd.to_datetime(resultado["fecha_alta"])
    fin_permanencia = pd.to_datetime(resultado["fin_permanencia"])

    # Contrato mensual: así lo indica tipo_contrato, o bien no tiene fecha de fin de permanencia
    es_mensual = (
        resultado["tipo_contrato"].astype(str).str.strip().str.lower().eq("mensual")
        | fin_permanencia.isna()
    )

    resultado["antiguedad_meses"] = (fecha_corte - fecha_alta).dt.days / DIAS_POR_MES

    meses_fin = (fin_permanencia - fecha_corte).dt.days / DIAS_POR_MES
    resultado["meses_hasta_fin_permanencia"] = np.where(es_mensual, 0.0, meses_fin)

    resultado["sin_permanencia_3m"] = (
        es_mensual | (resultado["meses_hasta_fin_permanencia"] <= 3)
    ).astype(int)

    return resultado
