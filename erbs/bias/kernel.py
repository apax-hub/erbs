import jax.numpy as jnp


def gaussian(g_diff, k, a):
    x = jnp.sum((g_diff) ** 2)
    U = k * jnp.exp(-x / (a * 2.0))
    return U
