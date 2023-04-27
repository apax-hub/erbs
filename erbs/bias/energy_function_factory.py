import jax.numpy as jnp
import numpy as np
from ase import units
from jax import vmap
from jraph import segment_mean
import jax

from erbs.bias.kernel import gaussian


def energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):
    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        bias = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

        return jnp.sum(bias)

    return energy_fn


class OPESExploreFactory:
    def __init__(self, T=300, dE=1.2, a=0.3) -> None:
        self.beta = 1 / (units.kB * T)
        self.std = a
        self.a = None
        self.k = None
        if dE < units.kB * T:
            raise ValueError("dE needs to be larger than 1.0!")
        self.dE = dE
        self.gamma = self.dE * self.beta

    def create(self, cv_fn, dim_reduction_fn, Z, g_ref, Z_ref, g_nl):
        n_atoms = Z.shape[0]
        cv_dim = g_ref.shape[-1]

        self.a = self.std ** (cv_dim * 2)
        self.k = 1 / (np.sqrt(2.0 * np.pi * self.std**2) **cv_dim )

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)

            prefactor = (self.gamma - 1.0) / self.beta
            eps = jnp.exp(-(self.dE / prefactor )/(n_atoms))
            
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE

            return total_bias

        return energy_fn


class MetaDCutFactory:
    def __init__(self, k=0.05, a=0.3, E_max=0.043) -> None:
        # self.beta = 1 / (units.kB * T)
        self.a = a
        self.k = k
        self.E_max = E_max

    def create(self, cv_fn, dim_reduction_fn, Z, g_ref, Z_ref, g_nl):
        n_atoms = Z.shape[0]
        cv_dim = g_ref.shape[-1]

        a = self.a ** (cv_dim * 2)
        k = self.k

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            bias_ij = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

            bias_i = jax.ops.segment_sum(bias_ij, g_nl[0], num_segments=n_atoms)

            bias_clipped = jnp.clip(bias_i, a_max=self.E_max)
            bias_cutoff = (self.E_max*n_atoms + A_min - A_curr)/n_atoms * jnp.sin(np.pi/2 * bias_clipped / self.E_max)
            bias = jnp.sum(bias_cutoff)
            bias_scaled_clipped = jnp.clip(bias, a_min=0.0)

            return bias_scaled_clipped

        return energy_fn