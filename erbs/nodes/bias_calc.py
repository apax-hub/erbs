import ase
from erbs.bias.energy_function_factory import OPESExploreFactory
from erbs.bias.potential import GKernelBias
from erbs.dim_reduction.elementwise_pca import ElementwiseLocalPCA
import zntrack
from ase import units
import ipsuite as ips
from typing import Optional

class ERBSCalculator(ips.base.IPSNode):
    _module_ = "erbs.nodes"

    model = zntrack.deps()
    data: Optional[list[ase.Atoms]] = zntrack.deps(None)

    n_basis: float = zntrack.params(4)
    r_min: float = zntrack.params(1.1)
    r_max = zntrack.params(6.0)
    n_contr: float = zntrack.params(8)

    pca_components: int = zntrack.params(2)
    barrier_factor = zntrack.params(3)
    band_width: float = zntrack.params(1.5)
    temperature = zntrack.params(300)
    bias_interval: int = zntrack.params(2000)
    initial_clusters: int = zntrack.params(5)
    nl_skin: float = zntrack.params(0.5)

    def run(self):
        pass

    def get_calculator(self, **kwargs):
        zpca = ElementwiseLocalPCA(self.pca_components, self.initial_clusters)

        dE = units.kB*self.temperature * self.barrier_factor
        energy_fn_factory = OPESExploreFactory(T=self.temperature, dE=dE, a=self.band_width)

        base_calc = self.model.get_calculator()
        calc = GKernelBias(
            base_calc,
            zpca,
            energy_fn_factory,
            n_basis = self.n_basis,
            r_min =self.r_min,
            r_max=self.r_max,
            dr_threshold=self.nl_skin,
            interval=self.bias_interval,
        )

        if self.data:
            calc.add_configs(self.data)

        return calc
