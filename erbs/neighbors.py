from typing import Tuple, Callable, Optional

import jax.numpy as jnp
import numpy as np
from apax.utils.jax_md_reduced import partition, space

def build_neighbor_fn(
    atoms,
    r_max: float,
    dr_threshold: float,
    batched: bool = False,
) -> Tuple[Optional[Callable], Optional[Callable], jnp.ndarray]:
    """
    Sets up the neighbor function and displacement function based on the atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to define the geometry.
    r_max : float
        Cutoff radius for the neighbor list.
    dr_threshold : float
        Skin distance for the neighbor list.
    batched : bool, default False
        If True, returns None for displacement_fn and neighbor_fn.

    Returns
    -------
    displacement_fn : Optional[Callable]
        JAX-MD displacement function.
    neighbor_fn : Optional[Callable]
        JAX-MD neighbor list function.
    box : jnp.ndarray
        Simulation box.
    """
    box = np.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if batched:
        displacement_fn = None
        neighbor_fn = None
    else:
        if np.all(box < 1e-6):
            displacement_fn, _ = space.free()
            frac_coords = False
        else:
            displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)
            frac_coords = True
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box,
            r_max,
            dr_threshold,
            fractional_coordinates=frac_coords,
            disable_cell_list=True,
            format=partition.Sparse,
        )
    return displacement_fn, neighbor_fn, box
