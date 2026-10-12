"""KNN y escalado estándar implementados desde cero.

Reglas:
- Los cálculos se hacen solo con NumPy. No se puede usar sklearn.neighbors, sklearn.preprocessing
  ni scipy.spatial.
- KNNClasificador hereda de ClassifierMixin y BaseEstimator solo para poder usarse dentro de un
  Pipeline y de GridSearchCV de scikit-learn.
- Nada de bucles sobre filas para calcular distancias: se calculan con operaciones vectorizadas,
  por lotes de `tamano_lote` filas para no llenar la memoria.

Verificá tu trabajo con:  python -m pytest -q tests/test_knn.py
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class EscaladorEstandar:
    """Estandariza cada columna: (x - media) / desvío.

    Atributos que crea fit: media_ y desvio_ (arrays de forma (n_columnas,)).
    - El desvío es el poblacional (np.std con ddof=0), igual que StandardScaler.
    - Si una columna es constante (desvío 0), se usa desvío 1 para no dividir por cero.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0, ddof=0)
        self.desvio_ = np.where(desvio == 0, 1.0, desvio)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.media_) / self.desvio_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNClasificador(ClassifierMixin, BaseEstimator):
    """Clasificador de k vecinos más cercanos (búsqueda por fuerza bruta).

    Parámetros
    ----------
    k : int, por defecto 5
        Cantidad de vecinos.
    pesos : "uniforme" o "distancia"
        "uniforme": cada vecino vota con peso 1.
        "distancia": cada vecino vota con peso 1 / distancia. Si la consulta coincide exactamente con
        uno o más ejemplos (distancia 0), votan solo esos ejemplos, con peso 1 cada uno.
    metrica : "euclidea" o "manhattan"
    tamano_lote : int, por defecto 1000
        Cantidad de filas de consulta que se procesan juntas al calcular distancias.

    Atributos que crea fit
    ----------------------
    X_ : array (n_entrenamiento, n_variables) con los datos de entrenamiento, en float.
    classes_ : array (n_clases,) con las clases ordenadas (np.unique).
    y_codificada_ : array (n_entrenamiento,) con el índice de la clase de cada ejemplo en classes_.
    """

    def __init__(self, k=5, pesos="uniforme", metrica="euclidea", tamano_lote=1000):
        self.k = k
        self.pesos = pesos
        self.metrica = metrica
        self.tamano_lote = tamano_lote

    def fit(self, X, y):
        """KNN no estima parámetros: guarda los datos. Devuelve self."""
        if self.metrica not in ("euclidea", "manhattan"):
            raise ValueError("metrica debe ser 'euclidea' o 'manhattan'")
        if self.pesos not in ("uniforme", "distancia"):
            raise ValueError("pesos debe ser 'uniforme' o 'distancia'")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X debe ser 2D y tener la misma cantidad de filas que y")
        if not 1 <= self.k <= len(X):
            raise ValueError("k debe estar entre 1 y la cantidad de ejemplos de entrenamiento")
        if self.tamano_lote < 1:
            raise ValueError("tamano_lote debe ser al menos 1")

        self.X_ = X
        self.classes_, codificada = np.unique(y, return_inverse=True)
        self.y_codificada_ = codificada.ravel()
        return self

    def _distancias(self, X):
        """Matriz (n_consultas, n_entrenamiento) con la distancia entre cada consulta y cada ejemplo.

        Pista para la euclídea: ||a - b||² = ||a||² + ||b||² - 2 a·b. Por redondeo puede dar valores
        levemente negativos: recortalos a 0 antes de la raíz cuadrada.
        """
        if self.metrica == "euclidea":
            cuadrados = (
                (X ** 2).sum(axis=1)[:, None]
                + (self.X_ ** 2).sum(axis=1)[None, :]
                - 2.0 * (X @ self.X_.T)
            )
            return np.sqrt(np.maximum(cuadrados, 0.0))

        # Manhattan: se acumula variable por variable; cada paso es vectorizado sobre todas las filas
        dist = np.zeros((X.shape[0], self.X_.shape[0]))
        for j in range(self.X_.shape[1]):
            dist += np.abs(X[:, j][:, None] - self.X_[:, j][None, :])
        return dist

    def kneighbors(self, X):
        """Devuelve (distancias, indices), ambos de forma (n_consultas, k), ordenados de menor a mayor distancia.

        Pista: np.argpartition encuentra los k menores sin ordenar toda la fila; después ordená solo esos k.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError("X debe tener la misma cantidad de columnas que el entrenamiento")

        distancias, indices = [], []
        for inicio in range(0, X.shape[0], self.tamano_lote):
            D = self._distancias(X[inicio:inicio + self.tamano_lote])
            idx = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]
            d = np.take_along_axis(D, idx, axis=1)
            orden = np.argsort(d, axis=1, kind="stable")
            distancias.append(np.take_along_axis(d, orden, axis=1))
            indices.append(np.take_along_axis(idx, orden, axis=1))

        return np.vstack(distancias), np.vstack(indices)

    def predict_proba(self, X):
        """Proporción (ponderada según `pesos`) de vecinos de cada clase. Columnas en el orden de classes_."""
        distancias, indices = self.kneighbors(X)

        if self.pesos == "uniforme":
            pesos = np.ones_like(distancias)
        else:
            with np.errstate(divide="ignore"):
                pesos = 1.0 / distancias
            # Filas con algún vecino a distancia 0: votan solo esos vecinos, con peso 1
            con_cero = (distancias == 0).any(axis=1)
            pesos[con_cero] = (distancias[con_cero] == 0).astype(float)

        clases_vecinos = self.y_codificada_[indices]
        votos = np.zeros((distancias.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            votos[:, c] = (pesos * (clases_vecinos == c)).sum(axis=1)

        return votos / votos.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Clase con mayor probabilidad. En caso de empate, la primera en classes_ (como np.argmax)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
