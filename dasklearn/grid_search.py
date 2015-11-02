from .core import fit, transform, predict, set_params

from functools import partial
import time
import numpy as np

from sklearn.cross_validation import _safe_split, check_scoring, check_cv, indexable
from sklearn.base import clone, is_classifier
from sklearn.grid_search import _CVScoreTuple, _check_param_grid, ParameterGrid

from dask.imperative import value, compute, do, Value
from dask.compatibility import apply

from .pipeline import Pipeline

do = partial(do, pure=True)


def _fit_and_score(estimator, X, y, scorer, train, test,
                   parameters, fit_params):
    if parameters is not None:
        estimator = set_params(estimator, **parameters)

    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    if y_train is None:
        estimator = estimator.fit(X_train, **fit_params)
    else:
        estimator = estimator.fit(X_train, y_train, **fit_params)

    test_score = estimator.score(X_test, y_test)

    ret = [test_score, X_test.shape[0], parameters]
    return ret


class BaseSearchCV(object):
    def __init__(self, estimator, scoring=None, fit_params=None, iid=True,
            refit=True, cv=None, verbose=0):

        self.scoring = scoring
        self.estimator = estimator
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose

    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        X, y = indexable(X, y)
        cv = check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))
        base_estimator = clone(self.estimator)

        best = best_parameters(base_estimator, cv, X, y, parameter_iterable,
                               self.scorer_, self.fit_params, self.iid)
        best = best.compute()

        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score


        if isinstance(base_estimator, Pipeline):
            base_estimator = base_estimator.to_sklearn().compute()

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = base_estimator.set_params(**best.parameters)
            if y is not None:
                self.best_estimator_ = best_estimator.fit(X, y, **self.fit_params)
            else:
                self.best_estimator_ = best_estimator.fit(X, **self.fit_params)
        return self

    def score(self, X, y=None):
        return self.best_estimator_.score(X, y)


def best_parameters(estimator, cv, X, y, parameter_iterable, scorer,
                    fit_params, iid):
    """ Lazily apply fit-and-score to data on all parameters / folds

    This function does little of the input checking and it doesn't trigger
    computation.

    Returns a lazy value object.  This should return almost immediately
    """
    _X, _y = X, y
    X = value(X)
    y = y if y is None else value(y)
    cv = [(value(train), value(test)) for train, test in cv]

    out = [_fit_and_score(estimator, X, y, scorer, train,
                          test, parameters, fit_params)
           for parameters in parameter_iterable
           for train, test in cv]

    return do(pick_best_parameters)(out, len(cv), iid)


class GridSearchCV(BaseSearchCV):
    """ Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : integer or cross-validation generator, default=3
        A cross-validation generator to use. If int, determines
        the number of folds in StratifiedKFold if estimator is a classifier
        and the target y is binary or multiclass, or the number
        of folds in KFold otherwise.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    Examples
    --------
    >>> from sklearn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svr = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svr, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape=None, degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params={}, iid=...,
           param_grid=..., refit=...,
           scoring=..., verbose=...)


    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a an hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
            iid=True, refit=True, cv=None, verbose=0):

        super(GridSearchCV, self).__init__(
            estimator, scoring, fit_params, iid,
            refit, cv, verbose)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        return self._fit(X, y, ParameterGrid(self.param_grid))


def pick_best_parameters(score_len_params, n_folds, iid):
    n_fits = len(score_len_params)

    scores = list()
    grid_scores = list()
    for grid_start in range(0, n_fits, n_folds):
        n_test_samples = 0
        score = 0
        all_scores = []
        for this_score, this_n_test_samples, parameters in \
                score_len_params[grid_start:grid_start + n_folds]:
            all_scores.append(this_score)
            if iid:
                this_score *= this_n_test_samples
                n_test_samples += this_n_test_samples
            score += this_score
        if iid:
            score /= float(n_test_samples)
        else:
            score /= float(n_folds)
        scores.append((score, parameters))
        grid_scores.append(_CVScoreTuple(
            parameters,
            score,
            np.array(all_scores)))

    # Find the best parameters by comparing on the mean validation score:
    # note that `sorted` is deterministic in the way it breaks ties
    best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
    return best
