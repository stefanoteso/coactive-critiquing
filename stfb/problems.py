# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import combinations
from sklearn.utils import check_random_state
from textwrap import dedent

def spnormal(n, sparsity=0.1, rng=None, dtype=None):
    """Samples from a 'sparse normal' distribution.

    First n * sparsity elements out of n are chosen to be the nonzeros. Then
    their value is sampled from a standard normal.
    """
    rng = check_random_state(rng)
    num_nonzeros = max(1, int(np.rint(n * sparsity)))
    nonzeros = rng.permutation(n)[:num_nonzeros]
    x = np.zeros(n, dtype=dtype)
    x[nonzeros] = rng.normal(0, 1, size=num_nonzeros)
    return x

def array_to_assignment(array, kind=None):
    assert array.ndim == 1
    if kind is None:
        kind = lambda x: x
    elif kind is bool:
        kind = lambda x: "true" if x else "false"
    return list(map(kind, array))

def assignment_to_array(assignment):
    array = np.zeros(len(assignment))
    for i, v in assignment.items():
        array[i - 1] = v
    return array

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

    def enumerate_features(self, features):
        """Computes the index set of all features, handling the 'all' and
        'attributes' cases."""
        if features == "attributes":
            return list(range(self.num_base_features))
        elif features == "all":
            return list(range(self.num_features))
        return features

    def get_feature_radius(self):
        """Returns the radius of a single feature."""
        raise NotImplementedError()

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
        x : numpy.ndarray of shape (num_features,)
            An optimal configuration.
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
        raise NotImplementedError()

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
        x_bar : numpy.ndarray of shape (num_features,)
            The locally optimal modification.
        """
        raise NotImplementedError()

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
        assert x.shape == (self.num_attributes,)
        w_star = self.w_star[self.enumerate_features(features)]
        x_star = self.x_star
        phi_star = self.phi(self.x_star, features)
        phi = self.phi(x, features)
        utility_star = w_star.dot(phi_star)
        utility = w_star.dot(phi)
#        print(dedent("""\
#            x*   = {x_star}
#            x    = {x}
#
#            phi* = {phi_star}
#            phi  = {phi}
#
#            u*   = {utility_star}
#            u    = {utility}
#        """).format(**locals()))
        assert utility <= utility_star, "utility loss is b0rked"
        return utility_star - utility
