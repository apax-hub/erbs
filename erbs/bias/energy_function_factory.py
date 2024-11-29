import jax.numpy as jnp
from ase import units
from jax import vmap

from erbs.bias.kernel import gaussian


class MetaDFactory:
    def __init__(self, T=300, dE=1.2, a=0.3, compression_threshold=0.4) -> None:
        self.std = a
        self.k = dE
        self.compression_threshold = compression_threshold

    def create(self, cv_fn, dim_reduction_fn):

        def energy_fn(positions, Z, neighbor, box, offsets, bias_state):

            g_ref = bias_state.g
            norm = bias_state.normalisation
            cov = bias_state.cov
            height = bias_state.height

            cv_dim = g_ref.shape[-1]

            g = cv_fn(positions, Z, neighbor.idx, box, offsets)
            g_reduced = dim_reduction_fn(g)

            g_diff = g_reduced - g_ref

            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.std)

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

            g_ref = bias_state.g
            norm = bias_state.normalisation
            cov = bias_state.cov
            height = bias_state.height

            cv_dim = g_ref.shape[-1]

            g = cv_fn(positions, Z, neighbor.idx, box, offsets)
            g_reduced = dim_reduction_fn(g)
            g_diff = g_reduced - g_ref

            g_diff = jnp.reshape(g_diff, (-1, cv_dim))
            kde_ij = vmap(gaussian, (0, 0, 0), 0)(g_diff, height, cov)

            unnormalized_prob_i = jnp.sum(kde_ij)
            prob_i = unnormalized_prob_i / norm

            prefactor = (self.gamma - 1.0) / self.beta

            eps = jnp.exp(-self.dE / prefactor)
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE

            return total_bias

        return energy_fn
