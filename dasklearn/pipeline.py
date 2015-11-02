from .core import fit, transform, predict, set_params
from sklearn.metrics import accuracy_score
import sklearn.pipeline
from toolz import groupby

from dask.imperative import do, value
from dask.base import normalize_token
from functools import partial
do = partial(do, pure=True)


class Pipeline(sklearn.pipeline.Pipeline):
    """ Dask version of sklearn.Pipeline

    This mimics the sklearn pipeline with the following changes

    1.  Methods are pure and side-effect free

        Before:  pipeline.fit(X)
        After:   pipeline = pipeline.fit(X)

    2.  Results are dask.imperative.Value objects.  Get the normal result by
        calling .compute() on them.  Or, alternatively, wait until you collect
        many and then collect them all at once.
    """
    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y=None):
        X = value(X)
        if y is not None:
            y = value(y)
        new_ests = []
        for name, est in self.steps:
            new_est = do(fit)(est, X, y)
            X = do(transform)(new_est, X)
            new_ests.append(new_est)

        return Pipeline([(name, new_est) for (name, old_est), new_est
                                          in zip(self.steps, new_ests)])

    def transform(self, X):
        for name, est in self.steps:
            X = do(transform)(est, X)
        return X

    def predict(self, X):
        for name, est in self.steps[:-1]:
            X = do(transform)(est, X)
        y = do(predict)(self.steps[-1][1], X)
        return y

    def score(self, X, y):
        X = value(X)
        y = value(y)
        y_predicted = self.predict(X)
        return do(accuracy_score)(y_predicted, y)

    def set_params(self, **params):
        d = groupby(0, [(k.split('__')[0], k.split('__', 1)[1], v)
                        for k, v in params.items()])
        d = {k: {a: b for _, a, b in v} for k, v in d.items()}
        steps = [(name, set_params(est, **d[name]) if name in d else est)
                 for name, est in self.steps]
        return Pipeline(steps)

    def to_sklearn(self):
        """ Create an sklearn pipeline object wrapped in a value

        >>> pipeline.to_sklearn().compute()  # doctest: +SKIP
        """
        return do(sklearn.pipeline.Pipeline)(self.steps)


@partial(normalize_token.register, sklearn.base.BaseEstimator)
def normalize_pipeline(est):
    return type(est), sorted(est.get_params().items())


@partial(normalize_token.register, Pipeline)
def normalize_pipeline(est):
    return type(est), est.steps
