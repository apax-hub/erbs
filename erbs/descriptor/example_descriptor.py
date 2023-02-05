
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import einops
import numpy as np
from jax_md import space


class GaussianBasis(hk.Module):
    def __init__(
        self, n_basis, r_min, r_max, dtype=jnp.float32, name: Optional[str] = None
    ):
        super().__init__(name)
        self.betta = n_basis**2 / r_max**2
        shifts = r_min + (r_max - r_min) / n_basis * np.arange(n_basis)

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        return basis


class RBFDescriptor(hk.Module):
    def __init__(
        self,
        displacement_fn,
        n_atoms,
        n_basis,
        r_min,
        r_max,
        dtype=jnp.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.n_atoms = n_atoms
        self.r_max = r_max
        self.radial_fn = GaussianBasis(
            n_basis,
            r_min,
            r_max,
            dtype=dtype,
            name="radial_fn",
        )

        self.displacement_fn = space.map_bond(displacement_fn)
        self.metric = space.map_bond(
            space.canonicalize_displacement_or_metric(displacement_fn)
        )
        self.dtype = dtype

    def __call__(self, R, Z, neighbor):
        R = R.astype(self.dtype)
        # R shape n_atoms x 3
        # Z shape n_atoms

        # dr shape: neighbors
        dr = self.metric(R[neighbor.idx[0]], R[neighbor.idx[1]])
        radial_basis = self.radial_fn(dr)
        descriptor = jax.ops.segment_sum(radial_basis, neighbor.idx[1], self.n_atoms)
        return descriptor
