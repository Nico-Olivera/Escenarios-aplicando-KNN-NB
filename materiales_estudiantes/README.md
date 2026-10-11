# Casos prácticos: Naive Bayes y KNN en problemas de empresa

Materiales para resolver los dos casos del enunciado.

| Carpeta | Caso | Algoritmo principal |
| --- | --- | --- |
| `caso1_cauce/` | Cauce: clasificación automática de mensajes de clientes | Naive Bayes multinomial |
| `caso2_nodo/` | Nodo Internet: predicción de bajas de clientes | KNN |

Cada carpeta tiene la misma estructura:

```
datos/        archivos CSV del caso
src/          módulos a completar (la especificación está en cada docstring)
tests/        pruebas automáticas: no se modifican
*.ipynb       notebook guía, con una sección por cada parte de las consignas
conftest.py   permite que pytest encuentre src/ (no se modifica)
```

## Preparar el entorno en la computadora

Se necesita Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para trabajar en un caso, abran el notebook desde su carpeta (los notebooks leen `datos/` e importan `src/` con rutas relativas):

```bash
cd caso1_cauce
jupyter lab caso1_cauce.ipynb
```

## Trabajar en Google Colab

1. Suban la carpeta completa de materiales a su Google Drive.
2. Abran el notebook del caso desde Drive con Google Colab.
3. Agreguen al principio del notebook una celda que monte Drive y se ubique en la carpeta del caso:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd "/content/drive/MyDrive/materiales_estudiantes/caso1_cauce"
```

Ajusten la ruta a la ubicación real de la carpeta. Colab ya trae instaladas todas las bibliotecas necesarias.

## Verificar las implementaciones

Desde la carpeta del caso:

```bash
python -m pytest -q                         # todas las pruebas del caso
python -m pytest -q tests/test_limpieza.py  # solo un módulo
```

Una función está terminada cuando pasan todas sus pruebas. Al principio fallan todas: es lo esperado.
