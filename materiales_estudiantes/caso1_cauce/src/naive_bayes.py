"""Naive Bayes multinomial implementado desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede importar scipy ni usar sklearn.naive_bayes.
- La clase hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn (eso le da get_params, set_params y score).
- Todo se calcula en escala logarítmica: multiplicar cientos de probabilidades chicas da 0
  en punto flotante (underflow).
- X puede ser un array de NumPy o una matriz dispersa de SciPy (la que devuelve CountVectorizer).
  Las operaciones X[filas], X.sum(axis=0) y X @ M funcionan con ambas; usá np.asarray(...) para
  convertir los resultados a arrays comunes.

Verificá tu trabajo con:  python -m pytest -q tests/test_naive_bayes.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def log_suma_exp(a, axis=1):
    """Calcula log(sum(exp(a))) a lo largo de `axis` sin overflow ni underflow.

    Truco: si m es el máximo de a, entonces log(sum(exp(a))) = m + log(sum(exp(a - m))).

    Ejemplo: log_suma_exp(np.array([[-1000.0, -1000.0]])) -> array([-999.3069...])
    """
    raise NotImplementedError("Completar log_suma_exp")


class NaiveBayesMultinomial(ClassifierMixin, BaseEstimator):
    """Naive Bayes multinomial con suavizado aditivo.

    Parámetros
    ----------
    alpha : float, por defecto 1.0
        Suavizado aditivo. Con alpha=1 es el suavizado de Laplace.

    Atributos que tiene que crear fit (mismos nombres que en scikit-learn)
    -------------------------------------------------------------------
    classes_ : array (n_clases,)
        Clases ordenadas, como devuelve np.unique(y).
    class_count_ : array (n_clases,)
        Cantidad de documentos de cada clase.
    feature_count_ : array (n_clases, n_terminos)
        Suma de los conteos de cada término en los documentos de cada clase.
    class_log_prior_ : array (n_clases,)
        log P(c) = log(documentos de la clase c / documentos totales).
    feature_log_prob_ : array (n_clases, n_terminos)
        log P(t | c) = log((conteo de t en c + alpha) / (total de términos en c + alpha * n_terminos)).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Estima las probabilidades a partir de la matriz de conteos X y las etiquetas y. Devuelve self."""
        raise NotImplementedError("Completar fit")

    def _log_verosimilitud_conjunta(self, X):
        """Devuelve log P(c) + sum_t x_t · log P(t | c) para cada documento y clase.

        Forma del resultado: (n_documentos, n_clases). Se resuelve con un solo producto matricial.
        """
        raise NotImplementedError("Completar _log_verosimilitud_conjunta")

    def predict_log_proba(self, X):
        """log P(c | documento): la log-verosimilitud conjunta normalizada con log_suma_exp."""
        raise NotImplementedError("Completar predict_log_proba")

    def predict_proba(self, X):
        """P(c | documento). Cada fila suma 1. Las columnas siguen el orden de classes_."""
        raise NotImplementedError("Completar predict_proba")

    def predict(self, X):
        """La clase con mayor probabilidad a posteriori para cada documento."""
        raise NotImplementedError("Completar predict")


def terminos_mas_indicativos(modelo, vocabulario, n=10):
    """Devuelve {clase: [n términos]} con los términos que más empujan hacia cada clase.

    Para la clase c, el puntaje de un término t es
        log P(t | c) - promedio de log P(t | c') en las demás clases c'.
    Se devuelven los n términos de mayor puntaje, de mayor a menor.

    `vocabulario` es la lista de términos en el orden de las columnas de X
    (por ejemplo, CountVectorizer.get_feature_names_out()).
    """
    raise NotImplementedError("Completar terminos_mas_indicativos")
