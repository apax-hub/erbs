import dataclasses
from typing import Optional
import jax.numpy as jnp
import numpy as np
from ase import units
from jax import Array, vmap
from erbs.cv.cv_nl import compute_cv_nl
from erbs.ops import segment_mean
import jax
from flax.struct import dataclass

from erbs.bias.kernel import gaussian


class MetaDFactory:
    def __init__(self, k=1.2, a=0.3) -> None:
        self.std = a
        self.k = k

    def create(self, cv_fn, dim_reduction_fn, cluster_models, Z, cluster_idxs, g_ref, cluster_idxs_ref, g_nl):
        Z = jnp.asarray(Z)
        cv_dim = g_ref.shape[-1]

        self.a = self.std ** (cv_dim * 2)

        cluster_idxs = jnp.asarray(cluster_idxs)

        def energy_fn(positions, neighbor):
            g = cv_fn(positions, Z, neighbor.idx)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, cluster_idxs)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            total_bias = jnp.sum(kde_ij)
            return total_bias

        return energy_fn


class OPESExploreFactory:
    def __init__(self, T=300, dE=1.2, a=0.3, compression_threshold=0.4) -> None:
        self.beta = 1 / (units.kB * T)
        self.std = a
        if dE < units.kB * T:
            raise ValueError("dE needs to be larger than 1.0!")
        self.dE = dE
        self.gamma = self.dE * self.beta
        self.compression_threshold = compression_threshold


    def create(self, cv_fn, dim_reduction_fn):

        def energy_fn(positions, Z, neighbor, box, offsets, bias_state):
            n_atoms = Z.shape[0]

            g_ref = bias_state.g
            # cluster_idxs = bias_state["cluster_idxs"]
            # g_nl = bias_state["feature_nl"]
            norm = bias_state.normalisation
            cov = bias_state.cov
            height = bias_state.height

            cv_dim = g_ref.shape[-1]
            # a = self.std ** (cv_dim * 2) # move to bias state
            # k = 1 / (np.sqrt(2.0 * np.pi * self.std**2) **cv_dim ) # move to bias state

            g = cv_fn(positions, Z, neighbor.idx, box, offsets)
            # g_reduced = vmap(dim_reduction_fn, 0, 0)(g)
            g_reduced = dim_reduction_fn(g)
            # n_ref = g_reduced.shape[0]
            # g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            g_diff = g_reduced - g_ref
            # print(g_reduced.shape)
            # print(g_ref.shape)
            # print(g_diff.shape)
            # print(height.shape)
            # print(cov.shape)
            # quit()
            g_diff = jnp.reshape(g_diff, (-1, cv_dim))
            kde_ij = vmap(gaussian, (0, 0, 0), 0)(g_diff, height, cov)

            # unnormalized_prob_i = jax.ops.segment_sum(kde_ij, g_nl[0], num_segments=n_atoms)
            unnormalized_prob_i = jnp.sum(kde_ij)
            prob_i = unnormalized_prob_i / norm

            prefactor = (self.gamma - 1.0) / self.beta

            eps = jnp.exp(-self.dE / prefactor)
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE

            return total_bias

        return energy_fn
