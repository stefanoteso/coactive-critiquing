# -*- encoding: utf-8 -*-

import numpy as np
from pymzn import minizinc
from itertools import combinations
from sklearn.utils import check_random_state
from textwrap import dedent

def _compute_sdep_mask(num_attributes, num_features, cliques, sparsity, rng):
    rng = check_random_state(rng)

    nnz_attributes = max(1, int(np.rint(num_attributes * sparsity)))
    nz_attributes = set(list(rng.permutation(num_attributes)[:nnz_attributes]))

    nz_features = []
    for j, clique in enumerate(cliques):
        if set(clique).issubset(nz_attributes):
            nz_features.append(j)

    return nz_features

def sdeptrinomial(num_attributes, num_features, cliques, sparsity, rng=None):
    """Samples from a {-1,0,1} uniform distribution with dependent features.

    First num_attributes * sparsity attributes are chosen as those that will
    be non-zero. The non-zero features are taken to be those that depend on
    the chosen non-zero attributes.

    The dependency structure determines how many non-zero features there
    will be.
    """
    nz_features = _compute_sdep_mask(num_attributes, num_features, cliques,
                                     sparsity, rng)
    x = np.zeros(num_features)
    x[nz_features] = 2 * rng.randint(0, 2, size=len(nz_features)) - 1
    return x

def sdepnormal(num_attributes, num_features, cliques, sparsity, rng=None):
    """Samples from a 'sparse normal' distribution with dependent features.

    First num_attributes * sparsity attributes are chosen as those that will
    be non-zero. The non-zero features are taken to be those that depend on
    the chosen non-zero attributes.

    The dependency structure determines how many non-zero features there
    will be.
    """
    nz_features = _compute_sdep_mask(num_attributes, num_features, cliques,
                                     sparsity, rng)
    x = np.zeros(num_features)
    x[nz_features] = rng.normal(0, 1, size=len(nz_features))
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
        assert self.x_star.shape == (num_attributes,), \
            "inference is b0rked: {} != {}".format(self.x_star.shape,
                                                   num_attributes)

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

    def phi(self, x, features, template_path, data):
        """Computes the feature representation of x.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The object.
        features : list or "all" or "attributes"
            The features to be used in the computation.
        template_path : str
            Path to the minizinc problem file.
        data : dict
            Data to be passed to the minizinc solver.

        Returns
        -------
        phi : numpy.ndarray of shape (num_features,)
            The feature representation of ``x``.
        """
        assert x.shape == (self.num_attributes,)

        assignments = minizinc(template_path, data=data, output_vars=["phi"],
                               keep=True)

        phi = self.assignment_to_array(assignments[0]["phi"])
        mask = np.ones_like(phi, dtype=bool)
        mask[self.enumerate_features(features)] = False
        phi[mask] = 0

        return phi

    def infer(self, w, features, template_path, data):
        """Searches for a maximum utility item.

        Parameters
        ----------
        w : numpy.ndarray of shape (num_features,)
            The weight vector.
        features : list or "all" or "attributes"
            The features to be used in the computation.
        template_path : str
            Path to the minizinc problem file.
        data : dict
            Data to be passed to the minizinc solver.

        Returns
        -------
        x : numpy.ndarray of shape (num_attributes,)
            An optimal configuration.
        """
        assert w.shape == (self.num_features,)

        targets = self.enumerate_features(features)
        if (w[targets] == 0).all():
            print("Warning: all-zero w!")

        assignments = minizinc(template_path, data=data,
                               output_vars=["x", "objective"],
                               keep=True)

        return self.assignment_to_array(assignments[0]["x"])

    def query_improvement(self, x, w_star, features, template_path, data):
        """Searches for a local maximum utility modification.

        If loss(x) is zero, i.e. x is optimal, it is return unmodified; under
        no other circumstances x can be returned unmodified.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        w_star : numpy.ndarray of shape (num_features,)
            The (possibly perturbet) user's latent weight vector.
        features : list or "all" or "attributes"
            The features to be used in the computation.
        template_path : str
            Path to the minizinc problem file.
        data : dict
            Data to be passed to the minizinc solver.

        Returns
        -------
        x_bar : numpy.ndarray of shape (n_attributes,)
            The locally optimal modification.
        """
        assert x.shape == (self.num_attributes,)

        if self.utility_loss(x, "all") == 0:
            # XXX this is noiseless
            return x

        targets = self.enumerate_features(features)
        assert (w_star[targets] != 0).any()

        assignments = minizinc(template_path, data=data,
                               output_vars=["x", "objective"],
                               keep=True)

        x_bar = self.assignment_to_array(assignments[0]["x"])
        assert (x != x_bar).any(), (x, x_bar)

        phi = self.phi(x, "all")
        phi_bar = self.phi(x_bar, "all")
        assert (phi != phi_bar).any()

        utility = np.dot(w_star, self.phi(x, targets))
        utility_bar = np.dot(w_star, self.phi(x_bar, targets))
        assert utility_bar > utility, \
            "u^k({}) = {} is not larger than u^k({}) = {}".format(
                x_bar, utility_bar, x, utility)

        return x_bar

    def query_critique(self, x, x_bar, features):
        """Searches for the maximum utility feature.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        x_bar : numpy.ndarray of shape (num_attributes,)
            The improved configuration.
        features : list or "all" or "attributes"
            The features to be used in the computation.

        Returns
        -------
        feature : int or None
            The locally optimal critique feature or None if no critique.
        """
        assert x.shape == (self.num_attributes,)
        assert x_bar.shape == (self.num_attributes,)
        assert (x != x_bar).any()

        targets = self.enumerate_features(features)
        assert len(targets) < self.num_features, \
            "requested critique in full feature space"

        scores = self.w_star * (self.phi(x_bar, "all") - self.phi(x, "all"))

        if self.noise == 0:
            scores[targets] = np.nan
            rho = np.nanargmax(scores)
        else:
            scores[targets] = 0
            if (scores != 0).any():
                pvals = scores / np.sum(scores)
                rho = np.nonzero(self.rng.multinomial(1, pvals))[0][0]
            else:
                # this can happen when x_bar is better than x only for features
                # that already belong to `targets`, in which case all other
                # features are equiprobable for critiquing purposes
                non_targets = list(set(range(self.num_features)) - set(targets))
                rho = non_targets[self.rng.randint(len(non_targets))]

        assert rho not in targets

        #return rho, -np.sign(scores[rho])
        return rho, np.sign(self.w_star[rho])

    def utility(self, x, features):
        """Computes the true utility of a configuration.

        Parameters
        ----------
        x : numpy.ndarray of shape (num_attributes,)
            The configuration.
        features : list or "all" or "attributes"
            The features to be used in the computation.
            The features to be used in the computation.

        Returns
        -------
        utility : float
            The utility of ``x``.
        """
        assert x.shape == (self.num_attributes,)

        targets = self.enumerate_features(features)
        phi = self.phi(x, features)

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
