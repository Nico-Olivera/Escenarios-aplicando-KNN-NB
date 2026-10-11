"""Pruebas de src/limpieza.py. Ejecutar desde la carpeta caso1_cauce con:  python -m pytest -q"""
import math

import pytest

from src.limpieza import limpiar_texto, quitar_tildes


def test_quitar_tildes():
    assert quitar_tildes("Ñandú pingüino acción") == "Nandu pinguino accion"


def test_no_texto_devuelve_vacio():
    assert limpiar_texto(None) == ""
    assert limpiar_texto(math.nan) == ""
    assert limpiar_texto(123) == ""


def test_minusculas_tildes_y_puntuacion():
    assert limpiar_texto("¡Hola! ¿Cómo están? NO PUEDO ENTRAR...") == "hola como estan no puedo entrar"


@pytest.mark.parametrize("entrada, esperado", [
    ("Me cobraron $ 15.000", "me cobraron _monto_"),
    ("me cobraron $15000 dos veces", "me cobraron _monto_ dos veces"),
    ("transferí $ 8.350,50 ayer", "transferi _monto_ ayer"),
    ("pagué Gs. 150.000 de luz", "pague _monto_ de luz"),
    ("son USD 20", "son _monto_"),
])
def test_montos(entrada, esperado):
    assert limpiar_texto(entrada) == esperado


@pytest.mark.parametrize("entrada, esperado", [
    ("mi DNI es 30.123.456", "mi dni es _num_"),
    ("hace 3 días", "hace _num_ dias"),
    ("operación N° 5543219", "operacion n _num_"),
])
def test_numeros(entrada, esperado):
    assert limpiar_texto(entrada) == esperado


def test_urls_y_correos():
    texto = "Entré a https://cauce-premios.xyz/login y a www.falso.com, escribí a soporte@cauce.com.ar"
    assert limpiar_texto(texto) == "entre a _url_ y a _url_ escribi a _email_"


def test_palabras_que_contienen_gs_no_son_montos():
    assert limpiar_texto("bugs 15") == "bugs _num_"


def test_emojis_y_espacios():
    assert limpiar_texto("  hola   😡😡  ayuda\n\n urgente!!! ") == "hola ayuda urgente"


def test_texto_vacio():
    assert limpiar_texto("") == ""
    assert limpiar_texto("👍") == ""
    assert limpiar_texto("...") == ""
