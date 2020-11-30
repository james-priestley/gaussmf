# from multiprocessing import Pool

import numpy as np
from numpy.random import default_rng

rng = default_rng()


class GaussianMFGenerator:

    """Helper class for sampling from a Gaussian Matrix Factorization model
    with random parameters.

    Parameters
    ----------
    n_features : int, optional
    n_components : int, optional
    sparsity : float, optional

    """

    def __init__(self, n_features=10, n_components=2, components_prior=1,
                 samples_prior=1):
        self.n_features = n_features
        self.n_components = n_components
        self.components_prior = components_prior
        self.samples_prior = samples_prior

        self._init_components()

    def _init_components(self):
        self.components_ = rng.normal(
            0, self.components_prior,
            size=(self.n_components, self.n_features),
        )

    def sample(self, n_samples):
        attributes = rng.normal(
            0, self.samples_prior, size=(self.n_components, n_samples)
        )

        X = default_rng().normal(self.components_.T @ attributes, 1)

        return X.T, attributes


class LDAGenerator:

    """Helper class for sampling from an LDA model with random parameters.

    Parameters
    ----------
    n_features : int, optional
        Number of words in the dictionary
    n_components : int, optional
        Number of topics
    alpha : float, optional
        Dirichlet prior on the per-document topic distributions
    eta : float, optional
        Dirichlet prior on the per-topic word distributions
    n_jobs : int, optional
        Number of workers to spawn in parallel pool, to parallelize document
        generation. Not implemented.
    """

    def __init__(self, n_features=100, n_components=10, alpha=None, eta=None,
                 n_jobs=1):

        self.n_features = n_features
        self.n_components = n_components
        self.alpha = alpha if alpha is not None else 1 / self.n_components
        self.eta = eta if eta is not None else 1 / self.n_components

        self._init_topics()

    def _init_topics(self):
        self.components_ = default_rng().dirichlet(
            [self.eta] * self.n_features,
            size=self.n_components)

    def _generate_topic_proportions(self):
        return default_rng().dirichlet([self.alpha] * self.n_components)

    def sample(self, n_samples, doc_length=None):

        if doc_length is None:
            doc_length = [self.n_features]

        X = np.zeros((n_samples, self.n_features))
        thetas = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            n = doc_length[0] if len(doc_length) == 1 else doc_length[i]

            # draw topic proportions for current document
            thetas[i] = self._generate_topic_proportions()

            # draw topics for each word and concatenate word distributions
            word_topics = np.random.choice(np.arange(self.n_components),
                                           size=n, p=thetas[i])
            X[i] = np.stack(
                [np.random.multinomial(1, p)
                 for p in self.components_[word_topics]]
                 ).sum(axis=0)

        return X, thetas
