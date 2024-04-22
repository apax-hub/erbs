from ase.io import read
import numpy as np
from ase.neighborlist import build_neighbor_list
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from xtb.ase.calculator import XTB

from ase.md.contour_exploration import ContourExploration

from erbs.transformations import repartition_hydrogen_mass

atoms = read("raw_data/ala2.xyz") # etoh.traj
atoms = repartition_hydrogen_mass(atoms, 2.0)

calc = XTB(method="gfn-ff")
atoms.calc = calc

T = 300
dt = 2.0
# duration = 10_000
friction=0.001
remaining_steps = 10_000 #int(np.ceil(duration / dt))

write_interval = 1
traj_path = f"test_data/ala2_ce_strong_5.traj"
MaxwellBoltzmannDistribution(atoms, temperature_K=T)
Stationary(atoms)
ZeroRotation(atoms)
# 
# dyn = VelocityVerlet(atoms, dt * units.fs)
# dyn = Langevin(atoms, dt * units.fs, friction=friction, temperature_K=T)
dyn = ContourExploration(atoms, energy_target=-92.9)

def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

# Now run the dynamics

traj = Trajectory(traj_path, 'w', atoms)
dyn.attach(printenergy, interval=500)
dyn.attach(traj.write, interval=write_interval)
dyn.run(remaining_steps)

