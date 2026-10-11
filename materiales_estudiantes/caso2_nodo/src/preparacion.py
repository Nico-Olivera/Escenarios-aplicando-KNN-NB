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
    raise NotImplementedError("Completar normalizar_metodo_pago")


def agregar_variables(df):
    """Devuelve una COPIA de df (sin modificar el original) con metodo_pago normalizado y tres columnas nuevas.

    Todas se calculan a la fecha_corte de cada fila, nunca a la fecha de hoy:
    - antiguedad_meses: días entre fecha_alta y fecha_corte, divididos por DIAS_POR_MES.
    - meses_hasta_fin_permanencia: días entre fecha_corte y fin_permanencia, divididos por DIAS_POR_MES.
      Vale 0 en los contratos mensuales (no tienen fecha de fin de permanencia).
    - sin_permanencia_3m: 1 si el cliente puede irse sin penalidad en los próximos 3 meses, es decir,
      si su contrato es mensual o si meses_hasta_fin_permanencia <= 3; 0 en otro caso (entero).
    """
    raise NotImplementedError("Completar agregar_variables")
