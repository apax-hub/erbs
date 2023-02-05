
import jax.numpy as jnp
import jax

def gauss(dr, k=0.1, a=0.1):
    # diff = jnp.mean((dr)**2) # rmsd, sqrt cancels with squaring in gauss
    diff = jnp.linalg.norm(dr, ord=2, axis=-1)
    U = k * jnp.exp(-diff /(a *2.0) )
    return U


def total_bias(g, ref_gs, k, a): # TODO add NL
    g_diff = g - ref_gs
    bias = jax.vmap(gauss, (0, None, None), 0)(g_diff, k, a)
    return jnp.sum(bias)