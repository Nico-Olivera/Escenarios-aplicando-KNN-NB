"""Pruebas de src/naive_bayes.py. Ejecutar desde la carpeta caso1_cauce con:  python -m pytest -q"""
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.naive_bayes import NaiveBayesMultinomial, log_suma_exp, terminos_mas_indicativos

CORPUS = [
    "no me llego la transferencia",
    "la transferencia sigue pendiente",
    "me rechazaron la tarjeta en el super",
    "no puedo activar la tarjeta",
    "me robaron el celular y hicieron transferencias",
    "compras con la tarjeta que no reconozco",
    "consumo que no reconozco me robaron",
]
ETIQUETAS = ["transferencias", "transferencias", "tarjeta", "tarjeta", "fraude", "fraude", "fraude"]


def datos_aleatorios(semilla=0, n=300, v=40, k=3):
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, k, n)
    tasas = rng.gamma(1.0, 1.0, (k, v))
    X = rng.poisson(tasas[y])
    return X, np.array(["a", "b", "c"])[y]


def test_atributos_y_formas():
    X, y = datos_aleatorios()
    m = NaiveBayesMultinomial().fit(X, y)
    assert list(m.classes_) == ["a", "b", "c"]
    assert m.class_count_.shape == (3,)
    assert m.feature_count_.shape == (3, 40)
    assert m.class_log_prior_.shape == (3,)
    assert m.feature_log_prob_.shape == (3, 40)


def test_calculo_a_mano():
    # 2 clases, 3 términos. Clase "x": documentos [2,1,0] y [0,1,0]. Clase "y": documento [0,0,3].
    X = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 3]])
    y = np.array(["x", "x", "y"])
    m = NaiveBayesMultinomial(alpha=1.0).fit(X, y)
    np.testing.assert_allclose(m.class_log_prior_, np.log([2 / 3, 1 / 3]))
    # x: conteos [2,2,0] + 1 = [3,3,1] / 7    y: conteos [0,0,3] + 1 = [1,1,4] / 6
    np.testing.assert_allclose(m.feature_log_prob_, np.log([[3 / 7, 3 / 7, 1 / 7], [1 / 6, 1 / 6, 4 / 6]]))


@pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0, 2.0])
def test_igual_que_scikit_learn(alpha):
    X, y = datos_aleatorios(semilla=1)
    Xtr, ytr, Xte = X[:200], y[:200], X[200:]
    propio = NaiveBayesMultinomial(alpha=alpha).fit(Xtr, ytr)
    ref = MultinomialNB(alpha=alpha).fit(Xtr, ytr)
    np.testing.assert_allclose(propio.class_log_prior_, ref.class_log_prior_)
    np.testing.assert_allclose(propio.feature_log_prob_, ref.feature_log_prob_)
    np.testing.assert_allclose(propio.predict_log_proba(Xte), ref.predict_log_proba(Xte), atol=1e-8)
    np.testing.assert_allclose(propio.predict_proba(Xte), ref.predict_proba(Xte), atol=1e-10)
    assert (propio.predict(Xte) == ref.predict(Xte)).all()


def test_acepta_matrices_dispersas():
    vec = CountVectorizer()
    X = vec.fit_transform(CORPUS)
    assert sparse.issparse(X)
    propio = NaiveBayesMultinomial(alpha=0.5).fit(X, ETIQUETAS)
    ref = MultinomialNB(alpha=0.5).fit(X, ETIQUETAS)
    nuevos = vec.transform(["me robaron la tarjeta", "no llego la transferencia"])
    np.testing.assert_allclose(propio.predict_proba(nuevos), ref.predict_proba(nuevos), atol=1e-10)
    assert list(propio.predict(nuevos)) == ["fraude", "transferencias"]


def test_probabilidades_validas_y_sin_underflow():
    X, y = datos_aleatorios(semilla=2)
    m = NaiveBayesMultinomial().fit(X, y)
    documento_enorme = X[:1] * 5_000  # decenas de miles de palabras: exp() de la suma daría 0 en todas las clases
    p = m.predict_proba(np.vstack([X[:50], documento_enorme]))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(m.predict_log_proba(documento_enorme)).all()


def test_log_suma_exp():
    from scipy.special import logsumexp
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (20, 5)) * np.array([1, 10, 100, 1_000, 10_000])
    np.testing.assert_allclose(log_suma_exp(a, axis=1), logsumexp(a, axis=1))
    np.testing.assert_allclose(log_suma_exp(np.array([[-1_000.0, -1_000.0]]), axis=1), [-1_000 + np.log(2)])


def test_la_matematica_no_usa_scipy():
    import inspect
    import re
    import src.naive_bayes as modulo
    codigo = inspect.getsource(modulo)
    assert not re.search(r"^\s*(import|from)\s+scipy", codigo, re.M), "Implementá los cálculos solo con NumPy"


def test_funciona_dentro_de_un_pipeline():
    pipe = Pipeline([("vec", CountVectorizer()), ("nb", NaiveBayesMultinomial())])
    pipe.fit(CORPUS, ETIQUETAS)
    assert pipe.predict(["no reconozco una compra"])[0] == "fraude"
    copia = clone(pipe)
    assert copia.get_params()["nb__alpha"] == 1.0
    busqueda = GridSearchCV(pipe, {"nb__alpha": [0.1, 1.0]}, cv=2).fit(CORPUS * 2, ETIQUETAS * 2)
    assert busqueda.best_params_["nb__alpha"] in (0.1, 1.0)


def test_terminos_mas_indicativos():
    vec = CountVectorizer()
    X = vec.fit_transform(CORPUS)
    m = NaiveBayesMultinomial().fit(X, ETIQUETAS)
    top = terminos_mas_indicativos(m, vec.get_feature_names_out(), n=3)
    assert set(top) == {"fraude", "tarjeta", "transferencias"}
    assert all(len(v) == 3 for v in top.values())
    assert "reconozco" in top["fraude"] and "robaron" in top["fraude"]
    assert "transferencia" in top["transferencias"]
