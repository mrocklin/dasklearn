from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFdr
from sklearn.datasets import make_blobs, fetch_20newsgroups
import sklearn.pipeline
import dask
import dasklearn as dl
import dask.imperative as di
from toolz import merge

X, y = make_blobs()


def test_pipeline():
    pipeline = dl.Pipeline([("scale", StandardScaler()),
                            ("fdr", SelectFdr()),
                            ("svm", LinearSVC())])

    pipeline = pipeline.fit(X, y)
    y2 = pipeline.predict(X)
    score = pipeline.score(X, y)

    assert isinstance(y2, di.Value)
    assert isinstance(score, di.Value)

    assert isinstance(score.compute(), float)

    assert pipeline.score(X, y).key == pipeline.score(X, y).key
    assert score.compute() == score.compute()

    y22 = y2.compute()
    assert y22.shape == y.shape
    assert y22.dtype == y.dtype
    skpipeline = sklearn.pipeline.Pipeline([("scale", StandardScaler()),
                                            ("fdr", SelectFdr()),
                                            ("svm", LinearSVC())])

    skpipeline.fit(X, y)
    sk_y2 = skpipeline.predict(X)
    sk_score = skpipeline.score(X, y)
    assert sk_score == score.compute()


def test_pipeline_shares_structure():
    pipeline = dl.Pipeline([("scale", StandardScaler()),
                            ("fdr", SelectFdr()),
                            ("svm", LinearSVC())])

    pipeline1 = pipeline.fit(X, y)
    score1 = pipeline1.score(X, y)

    pipeline2 = pipeline.set_params(svm__C=0.1)
    pipeline2 = pipeline2.fit(X, y)
    score2 = pipeline2.score(X, y)

    assert len(merge(score1.dask, score2.dask)) <= len(score1.dask) + 3
    assert score1.key != score2.key
