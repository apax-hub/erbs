import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from jax_md import partition, space

from erbs.cv.cv_nl import compute_cv_nl


def build_energy_neighbor_fns(atoms, r_max, dr_threshold):
    box = jnp.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if np.all(box < 1e-6):
        displacement_fn, _ = space.free()
        box = 100
    else:
        displacement_fn, _ = space.periodic(box)

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        r_max,
        dr_threshold,
        fractional_coordinates=False,
        format=partition.Sparse,
    )

    return neighbor_fn


class GKernelBias(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calc,
        cv_fn,
        dim_reduction_factory,
        energy_fn_factory,
        r_max=6.0,
        dr_threshold=0.5,
        interval=100,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        if not isinstance(base_calc, Calculator):
            raise ValueError(
                "All the calculators should be inherited from"
                "the ase's Calculator class"
            )
        self.base_calc = base_calc
        self.r_max = r_max
        self.dr_threshold = dr_threshold

        self.cv_fn = cv_fn
        self.dim_reduction_factory = dim_reduction_factory
        self.energy_fn_factory = energy_fn_factory

        self.energy_and_force_fn = None

        self.ref_cvs = []
        # self.reduced_ref_cvs = None
        # ^ maybe introduce if we allow for reduction function update interval
        self.ref_atomic_numbers = []
        self.neighbors = None
        self.neighbor_fn = None

        self.interval = interval
        self._step_counter = 0
        self.accumulate = True

    def _initialize_nl(self, atoms):
        self.neighbor_fn = build_energy_neighbor_fns(
            atoms, self.r_max, self.dr_threshold
        )

    def save_descriptors(self, path):
        np.savez(path, g=self.ref_cvs, z=self.ref_atomic_numbers)

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

        reduced_ref_cvs, sorted_ref_numbers = self.dim_reduction_factory.fit_transform(
            self.ref_cvs, self.ref_atomic_numbers
        )
        g_neighbors = compute_cv_nl(atoms.numbers, sorted_ref_numbers)

        energy_fn = self.energy_fn_factory.create(
            self.cv_fn,
            self.dim_reduction_factory.create_dim_reduction_fn(),
            numbers,
            reduced_ref_cvs,
            g_neighbors,
        )

        @jax.jit
        def body_fn(positions, neighbor):
            neighbor = neighbor.update(positions)
            energy, neg_forces = jax.value_and_grad(energy_fn)(positions, neighbor)
            forces = -neg_forces

            return {"energy": energy, "forces": forces}, neighbor

        self.energy_and_force_fn = body_fn

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.base_calc.calculate(atoms, properties, system_changes)
        self.base_results = self.base_calc.results

        if self._step_counter == 0:
            self.add_configs([atoms])
            self._initialize_nl(atoms)

        should_update_bias = self._step_counter % self.interval == 0
        if should_update_bias and self.accumulate:
            self.update_bias(atoms)

        position = jnp.array(atoms.positions, dtype=jnp.float32)
        bias_results, self.neighbors = self.energy_and_force_fn(
            position, self.neighbors
        )

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(position)
            bias_results, self.neighbors = self.energy_and_force_fn(
                position, self.neighbors
            )
        bias_results = {
            k: np.array(v, dtype=np.float64) for k, v in bias_results.items()
        }

        self.results = {
            "energy": self.base_results["energy"] + bias_results["energy"],
            "forces": self.base_results["forces"] + bias_results["forces"],
            "energy_label": self.base_results["energy"],
            "forces_label": self.base_results["forces"],
        }
        self._step_counter += 1

    def add_configs(self, atoms_list):
        @jax.jit
        def calc_g(positions, neighbors):
            neighbors = neighbors.update(positions)
            g = self.cv_fn(positions, neighbors)
            return g

        for atoms in atoms_list:
            positions = jnp.array(atoms.positions, dtype=jnp.float32)
            numbers = jnp.array(atoms.numbers, dtype=jnp.int32)

            if not self.neighbor_fn:
                self._initialize_nl(atoms)

            if not self.neighbors or self.neighbors.did_buffer_overflow:
                self.neighbors = self.neighbor_fn.allocate(positions)
            g = calc_g(positions, self.neighbors)
            self.ref_cvs.append(np.asarray(g))
            self.ref_atomic_numbers.append(np.asarray(numbers))
