from jax import vmap
import jax.numpy as jnp
import jax
from erbs.bias.kernel import gaussian
from ase import units
import numpy as np


def energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):

    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0,0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        bias = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)
        
        return jnp.sum(bias)

    return energy_fn


def opes_energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):
    T = 300
    beta = 1 / (units.kB * T) # 1/ everything eV/K * K -> eV
    k = 1 / (a * np.sqrt( 2.0 * np.pi))
    n_atoms = Z.shape[0]
    dE = 0.3

    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0,0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

        prob_i = jax.ops.segment_sum(kde_ij, g_nl[0], num_segments=n_atoms)

        gamma = (1 - (1/(beta*dE)) )
        eps = jnp.exp(- (beta * dE) / gamma )
        unscaled_bias = jnp.log( prob_i + eps )
        bias = gamma / beta * unscaled_bias

        return jnp.sum(bias)

    return energy_fn