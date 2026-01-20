import random

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator

# from ase.md.npt import NPT
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from tqdm import trange
from xtb.ase.calculator import XTB

from erbs.transformations import repartition_hydrogen_mass

# path = "raw_data/etoh.traj"
path = "raw_data/ala2.xyz"
# path = "raw_data/bmim_opt.extxyz"
atoms = read(path)
atoms = repartition_hydrogen_mass(atoms, 2.0)
# atoms.wrap()

T = 300
ts = 2.0
friction = 0.001
write_interval = 1
remaining_steps = 10_000
# max_steps_per_iteration = 1_000

traj_file = "test_data/nuts_nvt.extxyz"


calc = XTB(method="gfn-ff")  # GFN1-xTB gfn-ff

atoms.calc = calc


def run_traj(atoms, calc, ts, T, max_steps):
    del atoms.calc
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = Langevin(atoms, ts * units.fs, friction=friction, temperature_K=T)
    # dyn = VelocityVerlet(atoms, ts * units.fs)

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
        labeled_atoms.calc = SinglePointCalculator(labeled_atoms, **calc.results)
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
    remaining_steps -= successful_steps
