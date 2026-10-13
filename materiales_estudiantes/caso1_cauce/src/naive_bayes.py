"""Naive Bayes multinomial implementado desde cero usando NumPy."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def log_suma_exp(a, axis=1):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    resultado = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(resultado, axis=axis)


class NaiveBayesMultinomial(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        if self.alpha <= 0:
            raise ValueError("alpha debe ser mayor que cero")
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_clases = len(self.classes_)
        self.class_count_ = np.bincount(y_idx, minlength=n_clases).astype(float)
        n_terminos = X.shape[1]
        self.feature_count_ = np.zeros((n_clases, n_terminos), dtype=float)
        for i in range(n_clases):
            suma = X[y_idx == i].sum(axis=0)
            self.feature_count_[i] = np.asarray(suma).ravel()
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        suavizado = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(suavizado / suavizado.sum(axis=1, keepdims=True))
        self.n_features_in_ = n_terminos
        return self

    def _log_verosimilitud_conjunta(self, X):
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X):
        conjunta = self._log_verosimilitud_conjunta(X)
        return conjunta - log_suma_exp(conjunta, axis=1)[:, None]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_verosimilitud_conjunta(X), axis=1)]


def terminos_mas_indicativos(modelo, vocabulario, n=10):
    vocabulario = np.asarray(vocabulario)
    resultado = {}
    for i, clase in enumerate(modelo.classes_):
        otras = np.delete(modelo.feature_log_prob_, i, axis=0)
        referencia = otras.mean(axis=0) if len(otras) else np.zeros(modelo.feature_log_prob_.shape[1])
        puntajes = modelo.feature_log_prob_[i] - referencia
        indices = np.argsort(puntajes)[::-1][:n]
        resultado[clase] = vocabulario[indices].tolist()
    return resultado
