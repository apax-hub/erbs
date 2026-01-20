from functools import partial

import jax.numpy as jnp
import numpy as np
from apax.layers.descriptor import GaussianMomentDescriptor
from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from tqdm import trange
from xtb.ase.calculator import XTB

from erbs.bias import GKernelBias, OPESExploreFactory, energy_fn_factory
from erbs.dim_reduction import ElementwiseLocalPCA
from erbs.transformations import repartition_hydrogen_mass

# path = "raw_data/etoh.traj"
path = "raw_data/ala2.xyz"
# path = "raw_data/bmim_opt.extxyz"
atoms = read(path)
atoms = repartition_hydrogen_mass(atoms, 2.0)
# atoms.wrap()

box = jnp.asarray(atoms.get_cell().lengths())

r_max = 6.0

descriptor = GaussianMomentDescriptor(
    radial_fn=RadialFunction(
        4,
        basis_fn=GaussianBasis(
            n_basis=4,
            r_min=1.1,
            r_max=r_max,
        ),
        emb_init=None,
    ),
    n_contr=8,
)  # , Z=jnp.asarray(atoms.numbers)
descriptor_fn = partial(descriptor.apply, {}, box=None, offsets=None)

T = 300
ts = 2.0
friction = 0.001
# write_interval=1
remaining_steps = 10000
bias_interval = 2000
cache_size = 20

# name = "opes_n5_gm3_kb8_a06_pca3_300k"
# name = "elementwise_interval2000_a015_2"
name = "elementwise_interval2000_a015_300k_m2_2"
traj_file = f"talk_data/{name}.extxyz"

# g_path = "gmnn_test/opes_n5_gm3_kb4_a05_pca2_300k.npz"
# print(units.kB*T)
# print(dE / (units.kB*T))
# print(dE * len(atoms))

# print()
# quit()
# zpca = ElementwisePCA(2)
zpca = ElementwiseLocalPCA(3, 5)

dE = units.kB * T * 3  # / len(atoms)
# dE = 0.046 / len(atoms) - units.kB*T
energy_fn_factory = OPESExploreFactory(T=T, dE=dE, a=1.5)
# E_max=1.5/ 22
# energy_fn_factory = MetaDCutFactory(k=0.01, a=0.3, E_max=E_max)

xtb = XTB(method="gfn-ff")  # GFN1-xTB gfn-ff
calc = GKernelBias(
    xtb,
    descriptor_fn,
    zpca,
    energy_fn_factory,
    r_max=r_max,
    dr_threshold=0.5,
    interval=bias_interval,
)
atoms.calc = calc
# calc.add_configs([atoms]) # TODO remove the need for this!
# calc.load_descriptors(g_path)


MaxwellBoltzmannDistribution(atoms, temperature_K=T)
Stationary(atoms)
ZeroRotation(atoms)

dyn = Langevin(atoms, ts * units.fs, friction=friction, temperature_K=T)
# dyn = VelocityVerlet(atoms, ts * units.fs)
# print("bias limit", dE)

atoms_cache = []
bias = np.zeros(remaining_steps)
En = np.zeros(remaining_steps)
# for i in trange(0, remaining_steps):
with trange(0, remaining_steps, ncols=100, disable=False) as pbar:
    for i in range(0, remaining_steps):
        dyn.run(1)
        # quit()
        # atoms.wrap()

        labeled_atoms = atoms.copy()
        labeled_atoms.calc = SinglePointCalculator(labeled_atoms, **calc.base_results)
        # labeled_atoms.calc = SinglePointCalculator(labeled_atoms, **calc.results)
        energy_bias = calc.results["energy"] - calc.base_results["energy"]
        energy = calc.base_results["energy"]
        bias[i] = energy_bias
        En[i] = energy

        atoms_cache.append(labeled_atoms)
        if (i % cache_size) == 0:
            write(traj_file, atoms_cache, format="extxyz", append=True)
            atoms_cache = []
        pbar.set_postfix(E=f"{energy:.3f}, E_bias: {energy_bias:.3f}")
        pbar.update(1)
if len(atoms_cache) > 0:
    write(traj_file, atoms_cache, format="extxyz", append=True)
calc.save_descriptors(f"gmnn_test/{name}.npz")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)

ax[0].plot(En)
ax[0].plot(En + bias)
# ax[1].axhline(1, color="grey")
ax[1].plot(bias)
plt.savefig("traj7.png", dpi=300)
plt.show()


# traj = Trajectory(traj_path, 'w', atoms)
# dyn.attach(traj.write, interval=write_interval)
# dyn.run(remaining_steps)
# traj_path = "raw_data/test_ase.traj"
