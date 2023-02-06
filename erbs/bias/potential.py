
import jax.numpy as jnp
import numpy as np
import jax
from ase.calculators.calculator import Calculator, all_changes

from erbs.cv.cv_nl import compute_cv_nl


class GKernelMTD(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, cv_fn, dim_reduction_factory, energy_fn_factory, neighbor_fn, k, a, interval=100, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.k = k
        self.a = a
        
        self.cv_fn = cv_fn
        self.dim_reduction_factory = dim_reduction_factory
        self.energy_fn_factory = energy_fn_factory
        self.neighbor_fn = neighbor_fn
        
        self.energy_and_force_fn = None

        self.ref_cvs = []
        #self.reduced_ref_cvs = None
        # ^ maybe introduce if we allow for reduction function update interval
        self.ref_atomic_numbers = []
        self.neighbors = None

        self.interval = interval
        self._step_counter = 0
        self.accumulate = True

    def update_bias(self, atoms):
        position = jnp.array(atoms.positions, dtype=jnp.float32)
        numbers = jnp.array(atoms.numbers, dtype=jnp.int32)

        if self.neighbors is None:
            self.neighbors = self.neighbor_fn.allocate(position)
        else:
            self.neighbors = self.neighbors.update(position)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(position)        

        g_new = self.cv_fn(position, self.neighbors)
        self.ref_cvs.append(g_new)
        self.ref_atomic_numbers.append(atoms.numbers)
        
        reduced_ref_cvs, sorted_ref_numbers = self.dim_reduction_factory.fit_transform(self.ref_cvs, self.ref_atomic_numbers)
        g_neighbors = compute_cv_nl(atoms.numbers, sorted_ref_numbers)

        energy_fn = self.energy_fn_factory(
            self.cv_fn,
            self.dim_reduction_factory.create_dim_reduction_fn(),
            numbers,
            reduced_ref_cvs,
            g_neighbors,
            self.k,
            self.a
        )

        @jax.jit
        def body_fn(positions, neighbor):
            neighbor = neighbor.update(positions)
            energy, neg_forces = jax.value_and_grad(energy_fn)(positions, neighbor)
            forces = -neg_forces
            return energy, forces, neighbor

        self.energy_and_force_fn = body_fn

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        update_bias = self._step_counter % self.interval == 0
        if update_bias and self.accumulate:
            self.update_bias(atoms)
        
        position = jnp.array(atoms.positions, dtype=jnp.float32)
        energy, forces, self.neighbors = self.energy_and_force_fn(position, self.neighbors)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(position)
            energy, forces, self.neighbors = self.energy_and_force_fn(position, self.neighbors)

        self.results = {
            "energy": np.array(energy, dtype=np.float64),
            "forces": np.array(forces, dtype=np.float64)
        }
        self._step_counter += 1

    def save_descriptors(self, path):
        np.savez(path, g=self.ref_cvs, z=self.ref_atomic_numbers)

    def add_configs(self, atoms_list):

        @jax.jit
        def calc_g(positions, neighbors):
            neighbors = neighbors.update(positions)
            g = self.cv_fn(positions, neighbors)
            return g

        for atoms in atoms_list:
            positions = jnp.array(atoms.positions, dtype=jnp.float32)
            numbers = jnp.array(atoms.numbers, dtype=jnp.int32)

            if not self.neighbors or self.neighbors.did_buffer_overflow:
                self.neighbors = self.neighbor_fn.allocate(positions)
            g = calc_g(positions, self.neighbors)
            self.ref_cvs.append(np.asarray(g))
            self.ref_atomic_numbers.append(np.asarray(numbers))
