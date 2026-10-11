"""Pruebas de src/preparacion.py. Ejecutar desde la carpeta caso2_nodo con:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.preparacion import DIAS_POR_MES, agregar_variables, normalizar_metodo_pago


@pytest.mark.parametrize("entrada, esperado", [
    ("debito_automatico", "debito_automatico"),
    ("Débito automático", "debito_automatico"),
    ("DEBITO AUTOMATICO", "debito_automatico"),
    ("Debito_Automatico", "debito_automatico"),
    ("Tarjeta de crédito", "tarjeta_credito"),
    ("TARJETA CREDITO", "tarjeta_credito"),
    ("Tarjeta_Credito", "tarjeta_credito"),
    ("transferencia bancaria", "transferencia"),
    ("TRANSFERENCIA", "transferencia"),
    ("efectivo ", "efectivo"),
    ("EFECTIVO", "efectivo"),
])
def test_normalizar_metodo_pago(entrada, esperado):
    assert normalizar_metodo_pago(entrada) == esperado


def test_metodo_desconocido():
    with pytest.raises(ValueError):
        normalizar_metodo_pago("cheque")


def clientes():
    return pd.DataFrame({
        "id_cliente": ["A", "B", "C", "D"],
        "fecha_corte": ["2026-05-31", "2026-05-31", "2026-09-30", "2026-09-30"],
        "fecha_alta": ["2026-03-01", "2020-05-31", "2025-11-15", "2024-09-30"],
        "tipo_contrato": ["mensual", "12_meses", "12_meses", "24_meses"],
        "fin_permanencia": [None, "2026-07-15", "2026-11-15", "2026-09-30"],
        "metodo_pago": ["Efectivo", "debito_automatico", "TARJETA CREDITO", "transferencia bancaria"],
    })


def test_agregar_variables():
    original = clientes()
    d = agregar_variables(original)
    assert "antiguedad_meses" not in original.columns, "agregar_variables no debe modificar el DataFrame original"
    np.testing.assert_allclose(d["antiguedad_meses"], np.array([91, 2191, 319, 730]) / DIAS_POR_MES)
    np.testing.assert_allclose(d["meses_hasta_fin_permanencia"], np.array([0, 45, 46, 0]) / DIAS_POR_MES)
    assert list(d["sin_permanencia_3m"]) == [1, 1, 1, 1]
    assert list(d["metodo_pago"]) == ["efectivo", "debito_automatico", "tarjeta_credito", "transferencia"]


def test_permanencia_lejana():
    d = agregar_variables(clientes().assign(fin_permanencia=[None, "2027-05-31", "2026-12-31", "2026-12-30"]))
    np.testing.assert_allclose(d["meses_hasta_fin_permanencia"].iloc[1:], np.array([365, 92, 91]) / DIAS_POR_MES)
    assert list(d["sin_permanencia_3m"]) == [1, 0, 0, 1]
