import jax.numpy as jnp


def gaussian(g_diff, k=0.1, a=0.1):
    x = jnp.sum((g_diff) ** 2)  # rmsd, sqrt cancels with squaring in gauss
    U = k * jnp.exp(-x / (a * 2.0))
    return U
