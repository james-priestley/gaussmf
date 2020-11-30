import warnings

import numpy as np
from numpy.random import default_rng
from sklearn.base import BaseEstimator

rng = default_rng()


class GaussianMF(BaseEstimator):

    """Gaussian matrix factorization with an sklearn estimator API. Model
    fitting is done via coordinate ascent MAP inference.

    Parameters
    ----------
    n_components : int, optional

    Returns
    -------

    Attributes
    ----------
    components_ :
    log_joint_ :
    n_iter_ :
    """

    def __init__(self, n_components=2, max_iter=1000, tol=1e-2, step_size=1e-5,
                 components_prior=0.1, samples_prior=0.1):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.components_prior = components_prior
        self.samples_prior = samples_prior

    def _init_components(self, X):
        self.components_ = rng.normal(
            0, 0.1, size=(self.n_components, X.shape[1]))
        # self._scale_components()

    def _scale_components(self):
        self.components_ /= np.linalg.norm(
                self.components_, axis=1).reshape(-1, 1)

    def _update_components(self, X):
        # should we rescale them to unit norm every step?
        error = self._calc_elementwise_error(X)
        gradient = (-1 / self.components_prior) * self.components_ \
            + self._attributes @ error
        self.components_ += self.step_size * gradient
        # self._scale_components()

    def _init_attributes(self, X):
        # attributes matrix is private; it is overwritten during any call to
        # `self.transform`
        self._attributes = rng.normal(
            0, 0.1, size=(self.n_components, X.shape[0]))

    def _update_attributes(self, X):
        error = self._calc_elementwise_error(X)
        gradient = (-1 / self.samples_prior) * self._attributes \
            + self.components_ @ error.T
        self._attributes += self.step_size * gradient

    def _calc_elementwise_error(self, X, attributes=None):
        if attributes is None:
            attributes = self._attributes

        return X - attributes.T @ self.components_

    def reconstruction_error(self, X, attributes=None):
        """Sum of the element-wise squared error between X and the
        reconstruction given by the current matrix factorization."""
        return (self._calc_elementwise_error(
            X, attributes=attributes) ** 2).sum()

    def _log_proba(self, X, attributes=None):
        """Compute the log joint probability of the data and parameters"""
        if attributes is None:
            attributes = self._attributes

        log_prior_components = -1 / (2 * self.components_prior) \
            * (self.components_ ** 2).sum()
        log_prior_attributes = -1 / (2 * self.samples_prior) \
            * (attributes ** 2).sum()
        log_likelihood = - 1 / 2 * self.reconstruction_error(
            X, attributes=attributes)

        return log_prior_components + log_prior_attributes + log_likelihood

    def _fit(self, X, update_components=False):

        self._init_attributes(X)
        if update_components:
            self._init_components(X)

        log_joint = []
        n_iter = 0
        for i in range(self.max_iter):

            n_iter += 1

            if update_components:
                self._update_components(X)
            self._update_attributes(X)

            # assess log posterior and check convergence
            log_joint.append(self._log_proba(X))
            if i > 0:
                delta_score = np.diff(log_joint[-2:])
                if np.abs(delta_score) < self.tol:
                    break
                elif (i + 1 == self.max_iter):
                    warnings.warn(
                        "Convergence warning: maximum iterations "
                        + f"({self.max_iter}) reached."
                    )

        if update_components:
            self.log_joint_ = np.array(log_joint)
            self.n_iter_ = n_iter

        return self.components_, self._attributes

    def fit_transform(self, X):
        comp, attr = self._fit(X, update_components=True)
        return attr

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        """Hold components fixed and iteratively optimize attributes."""
        _, attr = self._fit(X, update_components=False)
        return attr

    def score(self, X, attributes=None):
        """Compute the squared reconstruction error of the data given the
        current components."""

        if attributes is None:
            attributes = self.transform(X)
        return self.reconstruction_error(X, attributes=attributes)
