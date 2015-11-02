from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import numpy as np

import dask
import dasklearn as dl
import dask.imperative as di
from dasklearn.grid_search import (best_parameters, check_scoring, check_cv,
        is_classifier)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)


def test_grid_search():
    pipeline = dl.Pipeline([("pca", PCA()),
                            ("select_k", SelectKBest()),
                            ("svm", LinearSVC())])
    param_grid = {'select_k__k': [1, 2, 3, 4],
                  'svm__C': np.logspace(-3, 2, 3)}
    grid = dl.GridSearchCV(pipeline, param_grid)

    with dask.set_options(get=dask.get):
        result = grid.fit(X_train, y_train).score(X_test, y_test)

    assert isinstance(result, float)


def test_best_parameters():
    pipeline = dl.Pipeline([("pca", PCA()),
                            ("select_k", SelectKBest()),
                            ("svm", LinearSVC())])
    param_grid = {'select_k__k': [1, 2, 3, 4],
                  'svm__C': np.logspace(-3, 2, 3)}
    grid = dl.GridSearchCV(pipeline, param_grid)
    parameter_iterable = [{'select_k__k': k, 'svm__C': x}
                         for k in [1, 2, 3, 4]
                         for x in np.logspace(-3, 2, 3)]

    cv = check_cv(grid.cv, X_train, y_train, classifier=is_classifier(pipeline))
    scorer = check_scoring(pipeline, scoring=grid.scoring)

    best = best_parameters(pipeline, cv, X_train, y_train,
                           parameter_iterable, scorer, grid.fit_params,
                           grid.iid)

    pipeline = pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert (len(best.dask)
          < len(cv) * len(parameter_iterable) * len(score.dask) / 2)
