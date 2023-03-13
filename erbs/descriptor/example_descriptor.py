from typing import Any, Callable

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space


class RBFDescriptorFlax(nn.Module):
    displacement_fn: Callable = space.free()[0]
    n_basis: int = 5
    r_min: float = 0.5
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        self.betta = self.n_basis**2 / self.r_max**2
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=self.dtype)

        self.metric = space.map_bond(
            space.canonicalize_displacement_or_metric(self.displacement_fn)
        )

    def __call__(self, R, neighbor):
        R = R.astype(self.dtype)
        # R shape n_atoms x 3
        n_atoms = R.shape[0]

        # dr shape: neighbors
        dr = self.metric(R[neighbor.idx[0]], R[neighbor.idx[1]])
        dr = einops.repeat(dr, "neighbors -> neighbors 1")

        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        radial_basis = jnp.exp(-self.betta * (distances**2))
        descriptor = jax.ops.segment_sum(radial_basis, neighbor.idx[1], n_atoms)
        return descriptor
