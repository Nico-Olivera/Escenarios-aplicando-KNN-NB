"""Limpieza de texto para los mensajes de clientes de Cauce."""
import re
import unicodedata


def quitar_tildes(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def limpiar_texto(texto) -> str:
    if not isinstance(texto, str):
        return ""
    texto = quitar_tildes(texto.lower())
    texto = re.sub(r"(?:https?://|www\.)\S+", " _url_ ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b", " _email_ ", texto, flags=re.IGNORECASE)
    numero = r"\d+(?:[.,]\d+)*"
    texto = re.sub(r"(?<!\w)(?:\$|gs\.?|usd)\s*" + numero + r"(?!\w)", " _monto_ ", texto, flags=re.IGNORECASE)
    texto = re.sub(numero, " _num_ ", texto)
    texto = re.sub(r"[^a-z_\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()
