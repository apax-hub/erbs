from jax_md.util import Array
import jax.numpy as jnp
import haiku as hk


def get_cv_model(
    DescriptorClass,
    displacement_fn,
    atomic_numbers: Array,
    **descriptor_kwargs
):

    Z = jnp.asarray(atomic_numbers, dtype=jnp.int32)

    @hk.without_apply_rng
    @hk.transform
    def model(R, neighbor):
        descriptor_fn = DescriptorClass(displacement_fn, **descriptor_kwargs)
        out = descriptor_fn(R, Z, neighbor)
        return out

    return model
