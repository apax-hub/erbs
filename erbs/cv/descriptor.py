from jax_md.util import Array
from jax_md import partition
from typing import Callable, List, Tuple
import jax.numpy as jnp
from jax_md import space
from functools import partial
import numpy as np
import haiku as hk
from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
)

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

def get_g_model(
    atomic_numbers: Array,
    displacement: DisplacementFn,
    box_size: float = 100.0,
    r_max: float = 6.0,
    n_basis: int = 7,
    n_radial: int = 5,
    dr_threshold: float = 0.5,
    nl_format: partition.NeighborListFormat = partition.Sparse,
    **neighbor_kwargs
) -> MDModel:

    neighbor_fn = partition.neighbor_list(
        displacement,
        box_size,
        r_max,
        dr_threshold,
        fractional_coordinates=False,
        format=nl_format,
        **neighbor_kwargs
    )

    n_atoms = atomic_numbers.shape[0]
    Z = jnp.asarray(atomic_numbers)
    # casting ot python int prevents n_species from becoming a tracer,
    # which causes issues in the NVT `apply_fn`
    n_species = int(np.max(Z) + 1)

    @hk.without_apply_rng
    @hk.transform
    def model(R, neighbor):
        descriptor = GaussianMomentDescriptor(
            displacement,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
            r_min=0.5,
            r_max=r_max,
        )
        out = descriptor(R, Z, neighbor)
        return out

    return neighbor_fn, model.init, model.apply


def build_descriptor_neighbor_fns(atoms, params, dr_threshold):
    atomic_numbers = jnp.asarray(atoms.numbers)
    # box = jnp.asarray(atoms.get_cell().lengths())

    displacement_fn, _ = space.free()

    neighbor_fn, _, descriptor = get_g_model(
        atomic_numbers=atomic_numbers,
        displacement_fn=displacement_fn,
        displacement=displacement_fn,
        dr_threshold=dr_threshold,
    )
    descriptor_fn = partial(descriptor, params)
    return descriptor_fn, neighbor_fn