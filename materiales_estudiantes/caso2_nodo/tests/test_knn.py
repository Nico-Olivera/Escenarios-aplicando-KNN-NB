"""Pruebas de src/knn.py. Ejecutar desde la carpeta caso2_nodo con:  python -m pytest -q"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.knn import EscaladorEstandar, KNNClasificador

METRICA_SK = {"euclidea": "euclidean", "manhattan": "manhattan"}
PESOS_SK = {"uniforme": "uniform", "distancia": "distance"}


def datos(semilla=0, n=400, d=6):
    rng = np.random.default_rng(semilla)
    X = rng.normal(0, 1, (n, d)) * rng.uniform(0.5, 50, d)
    y = (X[:, 0] / X[:, 0].std() + X[:, 1] / X[:, 1].std() + rng.normal(0, 1, n) > 0.8).astype(int)
    return X[:300], y[:300], X[300:], y[300:]


def test_escalador_igual_que_standard_scaler():
    Xtr, _, Xte, _ = datos()
    propio = EscaladorEstandar().fit(Xtr)
    ref = StandardScaler().fit(Xtr)
    np.testing.assert_allclose(propio.media_, ref.mean_)
    np.testing.assert_allclose(propio.desvio_, ref.scale_)
    np.testing.assert_allclose(propio.transform(Xte), ref.transform(Xte))
    np.testing.assert_allclose(EscaladorEstandar().fit_transform(Xtr), ref.transform(Xtr))


def test_escalador_columna_constante():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z = EscaladorEstandar().fit_transform(X)
    assert np.isfinite(Z).all()
    np.testing.assert_allclose(Z[:, 1], 0.0)


@pytest.mark.parametrize("metrica", ["euclidea", "manhattan"])
def test_vecinos_igual_que_scikit_learn(metrica):
    Xtr, ytr, Xte, _ = datos(semilla=1)
    propio = KNNClasificador(k=7, metrica=metrica).fit(Xtr, ytr)
    ref = NearestNeighbors(n_neighbors=7, algorithm="brute", metric=METRICA_SK[metrica]).fit(Xtr)
    d_propio, i_propio = propio.kneighbors(Xte)
    d_ref, i_ref = ref.kneighbors(Xte)
    assert d_propio.shape == (100, 7) and i_propio.shape == (100, 7)
    np.testing.assert_allclose(d_propio, d_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(i_propio, i_ref)


def test_distancias_ordenadas_y_no_negativas():
    Xtr, ytr, Xte, _ = datos(semilla=2)
    d, _ = KNNClasificador(k=10).fit(Xtr, ytr).kneighbors(Xte)
    assert (d >= 0).all()
    assert (np.diff(d, axis=1) >= 0).all()


@pytest.mark.parametrize("pesos", ["uniforme", "distancia"])
@pytest.mark.parametrize("k", [1, 5, 25])
def test_probabilidades_igual_que_scikit_learn(pesos, k):
    Xtr, ytr, Xte, _ = datos(semilla=3)
    propio = KNNClasificador(k=k, pesos=pesos).fit(Xtr, ytr)
    ref = KNeighborsClassifier(n_neighbors=k, weights=PESOS_SK[pesos], algorithm="brute").fit(Xtr, ytr)
    np.testing.assert_allclose(propio.predict_proba(Xte), ref.predict_proba(Xte), atol=1e-10)
    np.testing.assert_array_equal(propio.predict(Xte), ref.predict(Xte))


def test_etiquetas_de_texto():
    Xtr, ytr, Xte, _ = datos(semilla=4)
    etiquetas = np.where(ytr == 1, "se_va", "se_queda")
    m = KNNClasificador(k=5).fit(Xtr, etiquetas)
    assert list(m.classes_) == ["se_queda", "se_va"]
    assert set(m.predict(Xte)) <= {"se_queda", "se_va"}


def test_distancia_cero_con_pesos_por_distancia():
    # Un punto de consulta idéntico a un ejemplo de entrenamiento: ese vecino decide solo.
    Xtr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ytr = np.array([1, 0, 0, 0])
    m = KNNClasificador(k=3, pesos="distancia").fit(Xtr, ytr)
    p = m.predict_proba(np.array([[0.0, 0.0]]))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p, [[0.0, 1.0]])


def test_resultado_no_depende_del_lote():
    Xtr, ytr, Xte, _ = datos(semilla=5)
    chico = KNNClasificador(k=9, pesos="distancia", tamano_lote=7).fit(Xtr, ytr)
    grande = KNNClasificador(k=9, pesos="distancia", tamano_lote=10_000).fit(Xtr, ytr)
    np.testing.assert_allclose(chico.predict_proba(Xte), grande.predict_proba(Xte))


def test_funciona_dentro_de_un_pipeline():
    Xtr, ytr, Xte, yte = datos(semilla=6)
    pipe = Pipeline([("esc", StandardScaler()), ("knn", KNNClasificador(k=15))])
    pipe.fit(Xtr, ytr)
    assert pipe.score(Xte, yte) > 0.7
    assert clone(pipe).get_params()["knn__k"] == 15
    busqueda = GridSearchCV(pipe, {"knn__k": [5, 15], "knn__pesos": ["uniforme", "distancia"]}, cv=3).fit(Xtr, ytr)
    assert busqueda.best_params_["knn__k"] in (5, 15)
