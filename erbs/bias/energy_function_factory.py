from jax import vmap
import jax.numpy as jnp
import jax
from erbs.bias.kernel import gaussian
from ase import units
import numpy as np
from jraph import segment_mean

# TODO turn these functions into classes with k, a T etc as fields and factory as method
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
    beta = 1 / (units.kB * T)
    k = 1 / (a * np.sqrt( 2.0 * np.pi))
    n_atoms = Z.shape[0]
    dE = units.kB * T * 1.2

    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0,0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

        prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)

        gamma = (1 - (1/(beta*dE)) )
        eps = jnp.exp(- (beta * dE) / gamma )
        unscaled_bias_i = jnp.log( prob_i + eps )
        bias_i = gamma / beta * unscaled_bias_i

        return jnp.sum(bias_i)

    return energy_fn