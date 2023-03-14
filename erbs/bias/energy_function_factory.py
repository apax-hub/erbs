import jax.numpy as jnp
import numpy as np
from ase import units
from jax import vmap
from jraph import segment_mean

from erbs.bias.kernel import gaussian


def energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):
    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        bias = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

        return jnp.sum(bias)

    return energy_fn



class OPESFactory:
    def __init__(self, T=300, dE=1.2, a=0.3) -> None:
        self.beta = 1 / (units.kB * T)
        self.a = a
        self.k = 1 / (a * np.sqrt(2.0 * np.pi))
        if dE < units.kB * T:
            raise ValueError("dE needs to be larger than 1.0!")
        self.dE = dE = units.kB * T * dE

    def create(self, cv_fn, dim_reduction_fn, Z, g_ref, g_nl):
        n_atoms = Z.shape[0]

        def energy_fn(positions, neighbor):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)

            gamma = 1 - (1 / (self.beta * self.dE)) 
            # This should be (gamma - 1) instead of (1-1/gamma) for opes explore
            eps = jnp.exp(-(self.beta * self.dE) / gamma)
            unscaled_bias_i = jnp.log(prob_i + eps)
            bias_i = gamma / self.beta * unscaled_bias_i

            return jnp.sum(bias_i)

        return energy_fn
