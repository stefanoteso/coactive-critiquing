# -*- encoding: utf-8 -*-

import numpy as np
from itertools import combinations
from sklearn.utils import check_random_state
from textwrap import dedent

def sdepnormal(num_attributes, num_features, deps, sparsity=0.1, rng=None,
               dtype=None):
    """Samples from a 'sparse normal' distribution with dependent features.

    First num_attributes * sparsity attributes are chosen as those that will
    be non-zero. The non-zero features are taken to be those that depend on
    the chosen non-zero attributes.

    The dependency structure determines how many non-zero features there
    will be.
    """
    rng = check_random_state(rng)

    nnz_attributes = max(1, int(np.rint(num_attributes * sparsity)))
    nz_attributes = set(list(rng.permutation(num_attributes)[:nnz_attributes]))

    nz_features = []
    for j, clique in deps:
        if set(clique) & nz_attributes:
            nz_features.append(j)

    x = np.zeros(num_features, dtype=dtype)
    x[nz_features] = rng.normal(0, 1, size=(len(nz_features)))
    return x

class Problem(object):
    """Base class for all problems.

    Attributes
    ----------
    x_star : numpy.ndarray
        An optimal configuration.

    Parameters
    ----------
    num_attributes : positive int
        The number of attributes (trivial features)
    num_base_features : positive int
        The number of attribute-level features
    num_features : positive int
        The number of features
    w_star : numpy.ndarray of shape (num_features,)
        The optimal, latent weight vector.
    """
    def __init__(self, num_attributes, num_base_features, num_features, w_star):
        if not (0 < num_attributes):
            raise ValueError("invalid number of attributes")
        if not (0 < num_base_features <= num_features):
            raise ValueError("invalid number of features")
        if w_star.shape != (num_features,):
            raise ValueError("mismatching w_star")

        self.num_attributes = num_attributes
        self.num_base_features = num_base_features
        self.num_features = num_features
        self.w_star = w_star

        self.x_star = self.infer(self.w_star, "all")
        assert self.x_star.shape == (num_attributes,), "inference is b0rked: {} != {}".format(self.x_star.shape, num_attributes)

    @staticmethod
    def array_to_assignment(array, kind=None):
        assert array.ndim == 1
        if kind is None:
            kind = lambda x: x
        elif kind is bool:
            kind = lambda x: "true" if x else "false"
        return list(map(kind, array))

    @staticmethod
    def assignment_to_array(assignment):
        array = np.zeros(len(assignment))
        for i, v in enumerate(assignment):
            array[i] = v
        return array

    def get_feature_radius(self):
        """Returns the radius of a single feature."""
        raise NotImplementedError()

    def enumerate_features(self, features):
        """Computes the index set of all features, handling the 'all' and
        'attributes' cases."""
        if features == "attributes":
            return list(range(self.num_base_features))
        elif features == "all":
            return list(range(self.num_features))
        return features

    def phi(self, x, features):
        """Computes the feature representation of x.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The object.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        phi : numpy.ndarray of shape (num_features,)
            The feature representation of ``x``.
        """
        raise NotImplementedError()

    def infer(self, w, features):
        """Searches for a maximum utility item.

        Parameters
        ----------
        w : numpy.ndarray of shape (num_features,)
            The weight vector.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        x : numpy.ndarray of shape (num_attributes,)
            An optimal configuration.
        """
        raise NotImplementedError()

    def compute_best_configuration(self, features):
        proj_w_star = self.w_star[self.enumerate_features(features)]
        return self.infer(proj_w_star, features)

    def query_improvement(self, x, features):
        """Searches for a local maximum utility modification.

        If x is optimal, it may be returned unmodified.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        x_bar : numpy.ndarray of shape (n_attributes,)
            The locally optimal modification.
        """
        raise NotImplementedError()

    def query_critique(self, x, features):
        """Searches for the maximum utility feature.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        feature : int or None
            The locally optimal critique feature or None if no critique.
        """
        assert x.shape == (self.num_attributes,)

        x_bar = self.query_improvement(x, features)

        u = self.utility(x, "all")
        u_bar = self.utility(x_bar, "all")
        u_star = self.utility(self.x_star, "all")

        if (u_bar - u) >= 0.1 * (u_star - u):
            return None, None

        targets = self.enumerate_features(features)

        scores = self.w_star * self.phi(x, "all")
        scores[targets] = np.nan
        try:
            rho = np.nanargmin(scores)
        except ValueError:
            return None, None

        sign = np.sign(self.w_star[rho])

        return rho, sign

    def utility(self, x, features):
        """Computes the utility of a configuration."""
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)
        phi = self.phi(x, targets)

        return np.dot(self.w_star[targets], phi[targets])

    def utility_loss(self, x, features):
        """Computes the utility loss.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        loss : non-negative float
            The utility loss of ``x``.
        """
        return self.utility(self.x_star, features) - self.utility(x, features)
