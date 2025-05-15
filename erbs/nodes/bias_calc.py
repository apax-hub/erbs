from typing import Optional

import ase
import ipsuite as ips
import numpy as np
import zntrack
from ase import units

from erbs.bias.energy_function_factory import OPESExploreFactory
from erbs.bias.potential import ERBS

# from erbs.dim_reduction.elementwise_pca import ElementwiseLocalPCA
from erbs.dim_reduction.elementwise_pca import GlobalPCA


class ERBSCalculator(ips.base.IPSNode):
    _module_ = "erbs.nodes"

    model = zntrack.deps()
    data: Optional[list[ase.Atoms]] = zntrack.deps(None)
    data_for_dimred_only: bool = zntrack.params(True)

    n_basis: float = zntrack.params(4)
    r_min: float = zntrack.params(1.1)
    r_max = zntrack.params(6.0)
    n_contr: float = zntrack.params(8)

    pca_components: int = zntrack.params(5)
    skip_first_n_components: Optional[int] = zntrack.params(None)

    barrier_factor = zntrack.params(3)
    band_width: float = zntrack.params(1.5)
    temperature = zntrack.params(300)
    bias_interval: int = zntrack.params(2000)
    nl_skin: float = zntrack.params(0.5)
    update_iterations: int = zntrack.params(np.inf)

    def run(self):
        pass

    def get_calculator(self, **kwargs):
        # zpca = ElementwiseLocalPCA(self.pca_components, self.initial_clusters)
        pca = GlobalPCA(
            self.pca_components, skip_first_n_components=self.skip_first_n_components
        )

        dE = units.kB * self.temperature * self.barrier_factor
        energy_fn_factory = OPESExploreFactory(
            T=self.temperature, dE=dE, a=self.band_width
        )

        base_calc = self.model.get_calculator()
        calc = ERBS(
            base_calc,
            pca,
            energy_fn_factory,
            n_basis=self.n_basis,
            r_min=self.r_min,
            r_max=self.r_max,
            dr_threshold=self.nl_skin,
            interval=self.bias_interval,
        )

        if self.data:
            calc.add_configs(self.data, for_dimred_only=self.data_for_dimred_only)

        return calc
