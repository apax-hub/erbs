from jax import vmap
import jax.numpy as jnp
from erbs.bias.kernel import gaussian


def energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):

    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0,0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        bias = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)
        
        return jnp.sum(bias)

    return energy_fn