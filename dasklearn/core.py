from sklearn.base import clone
from dask.imperative import do, value
from functools import partial

do = partial(do, pure=True)


def fit(est, X, y=None):
    est = clone(est)
    est.fit(X, y)
    return est


def transform(est, X):
    return est.transform(X)


def predict(est, X):
    return est.predict(X)


def set_params(est, **params):
    est = clone(est)
    return est.set_params(**params)


from dask.base import normalize_token
import sklearn.base

@partial(normalize_token.register, sklearn.base.BaseEstimator)
def normalize_estimator(est):
    return type(est), est.get_params()
