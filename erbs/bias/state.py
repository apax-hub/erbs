import dataclasses
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from jax import Array

from erbs.bias.kernel import compress, global_mc_normalisation, incremental_compress


@dataclass
class BiasState:
    std: float
    compression_threshold: float

    g: Optional[Array] = None
    height: Optional[Array] = None
    cov: Optional[Array] = None
    normalisation: Optional[Array] = None

    def replace(self, **kwargs) -> "BiasState":
        return dataclasses.replace(self, **kwargs)

    def initialize(self):
        cv_dim = self.g.shape[-1]
        # TODO use determinant here somewhere
        # k0 = self.std ** (cv_dim * 2)
        k0 = 1 / (np.sqrt(2.0 * np.pi * self.std**2) ** cv_dim)
        height = jnp.full((self.g.shape[0], 1), k0)

        cov = jnp.full(self.g.shape, self.std)

        Zn = global_mc_normalisation(
            np.asarray(self.g),
            np.asarray(height),
            np.asarray(cov),
        )

        return self.replace(height=height, cov=cov, normalisation=Zn)

    def compress(self):
        g, cov, height = compress(
            self.g, self.cov, self.height, thresh=self.compression_threshold
        )

        return self.replace(g=g, cov=cov, height=height)

    def incremental_compress(self, gnew):
        cv_dim = self.g.shape[-1]
        covnew = self.std ** (cv_dim * 2)  # move to bias state
        hnew = 1 / (np.sqrt(2.0 * np.pi * self.std**2) ** cv_dim)

        g, cov, height = incremental_compress(
            self.g,
            self.cov,
            self.height,
            gnew,
            covnew,
            hnew,
            thresh=self.compression_threshold,
        )

        return self.replace(g=g, cov=cov, height=height)

    def add_configuration(self, gnew):
        cv_dim = self.g.shape[-1]
        covnew = np.full((cv_dim), self.std)  # move to bias state
        hnew = 1 / (np.sqrt(2.0 * np.pi * self.std**2) ** cv_dim)
        hnew = np.array([hnew])

        g, cov, height = incremental_compress(
            np.array(self.g),
            np.array(self.cov),
            np.array(self.height),
            np.array(gnew),
            covnew,
            hnew,
            thresh=self.compression_threshold,
        )

        # for Zn: when compressing kernels, just compute the uncompressed contribution
        # to the normalisation

        # Zn = incremental_mc_normalisation()
        Zn = global_mc_normalisation(
            g,
            height,
            cov,
        )

        new_state = self.replace(
            g=jnp.array(g), cov=jnp.array(cov), height=jnp.array(height), normalisation=Zn
        )

        return new_state
