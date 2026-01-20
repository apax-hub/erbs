import random
from functools import partial

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator

# from ase.md.npt import NPT
from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from tqdm import trange
from xtb.ase.calculator import XTB

from erbs.bias import GKernelBias, OPESExploreFactory
from erbs.descriptor import RBFDescriptorFlax
from erbs.dim_reduction import ElementwisePCA
from erbs.transformations import repartition_hydrogen_mass


def setup_ase():
    """Add uncertainty keys to ASE all properties."""
    from ase.calculators.calculator import all_properties

    for val in ["forces_biased", "energy_biased"]:
        if val not in all_properties:
            all_properties.append(val)


setup_ase()

# path = "raw_data/etoh.traj"
path = "raw_data/ala2.xyz"
# path = "raw_data/bmim_opt.extxyz"
atoms = read(path)
atoms = repartition_hydrogen_mass(atoms, 2.0)
# atoms.wrap()

T = 500
ts = 2.0
friction = 0.001
write_interval = 1
remaining_steps = 10_000
# max_steps_per_iteration = 1_000

traj_file = "test_data/nuts_bias_opes_500k_4b_3d_a05_rescale_rand_biasedplacement.extxyz"


xtb = XTB(method="gfn-ff")  # GFN1-xTB gfn-ff


r_max = 5.0

descriptor = RBFDescriptorFlax(r_max=r_max)
descriptor_fn = partial(descriptor.apply, {})
zpca = ElementwisePCA(3)
# E_max=1.6/ 22
# energy_fn_factory = MetaDCutFactory(k=0.02, a=0.2, E_max=E_max)

dE = units.kB * T * 4  # / len(atoms)
# dE = 0.046 / len(atoms) - units.kB*T
energy_fn_factory = OPESExploreFactory(T=T, dE=dE, a=0.5)

calc = GKernelBias(
    xtb,
    descriptor_fn,
    zpca,
    energy_fn_factory,
    r_max=r_max,
    dr_threshold=0.5,
    interval=100,
)
calc.accumulate = False

atoms.calc = calc
calc._initialize_nl(atoms)
calc.add_configs([atoms])
calc.update_bias(atoms)


def run_traj(atoms, calc, ts, T, max_steps):
    del atoms.calc
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)
    ZeroRotation(atoms)

    # dyn = Langevin(atoms, ts * units.fs, friction=friction, temperature_K=T)
    # dyn = VelocityVerlet(atoms, ts * units.fs)
    dyn = NVTBerendsen(
        atoms, ts * units.fs, temperature_K=T, taut=0.005 * 1000 * units.fs
    )

    theta0 = np.reshape(atoms.get_positions(), (-1,))

    atoms_cache = []
    # conditions = np.zeros(max_steps)
    for i in trange(1, max_steps + 1, leave=False):
        dyn.run(1)

        # atoms.wrap()

        theta = np.reshape(atoms.get_positions(), (-1,))
        d_theta = theta - theta0
        momenta = np.reshape(atoms.get_momenta(), (-1,))
        condition = np.dot(d_theta, momenta)

        labeled_atoms = atoms.copy()
        labeled_atoms.calc = SinglePointCalculator(
            labeled_atoms, energy_biased=calc.results["energy"], **calc.base_results
        )
        atoms_cache.append(labeled_atoms)

        if condition < 0.0:
            break

    return atoms_cache, i


while remaining_steps > 0:
    # print(remaining_steps)
    atoms_cache, successful_steps = run_traj(atoms, calc, ts, T, remaining_steps)

    if len(atoms_cache) > 0:
        write(traj_file, atoms_cache, format="extxyz", append=True)

    atoms = random.choice(atoms_cache)
    energies = [a.calc.results["energy_biased"] for a in atoms_cache]  # _biased
    idx_of_min = np.argmin(energies)
    E_min_atoms = atoms_cache[idx_of_min]
    # atoms = E_min_atoms
    calc.update_bias(E_min_atoms)
    remaining_steps -= successful_steps
