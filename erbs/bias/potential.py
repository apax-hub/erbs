from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from jax_md import partition, space

from erbs.cv.cv_nl import compute_cv_nl


def append_to_ds(ds, new_data, batch_dim=True):
    shape = ds.shape[0] + new_data.shape[0]
    ds.resize(shape, axis=0)
    if batch_dim:
        ds[-new_data.shape[0], ...] = new_data
    else:
        ds[-new_data.shape[0] :, ...] = new_data


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
        log_file="labels.hdf5",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        if not isinstance(base_calc, Calculator):
            raise ValueError(
                "All the calculators should be inherited from"
                "the ase's Calculator class"
            )
        self.base_calc = base_calc
        self.log_file = log_file
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
        self.data_h5 = h5py.File(self.log_file, "w", libver="latest")
        self.initialized = False

    def _initialize_nl(self, atoms):
        self.neighbor_fn = build_energy_neighbor_fns(
            atoms, self.r_max, self.dr_threshold
        )

    def _initialize_g_ds(self):
        g_grp = self.data_h5.create_group("descriptors")
        cv_shape = self.ref_cvs[0].shape
        g_grp.create_dataset(
            "full", data=self.ref_cvs[0], maxshape=(None, cv_shape[1]), dtype=np.float32
        )
        g_grp.create_dataset(
            "atomic_numbers",
            data=self.ref_atomic_numbers[0],
            maxshape=(None,),
            dtype=np.int32,
        )

    def _initialize_label_ds(self):
        label_grp = self.data_h5.create_group("labels")
        E_base = np.array([self.base_results["energy"]])
        E_bias = np.array([self.bias_results["energy"]])

        label_grp.create_dataset(
            "energy", data=E_base, maxshape=(None,), dtype=np.float64
        )
        label_grp.create_dataset(
            "energy_bias", data=E_bias, maxshape=(None,), dtype=np.float64
        )

        n_atoms = self.base_results["forces"].shape[0]
        F_base = self.base_results["forces"][None, ...]
        F_bias = self.bias_results["forces"][None, ...]
        label_grp.create_dataset(
            "forces", data=F_base, maxshape=(None, n_atoms, 3), dtype=np.float64
        )
        label_grp.create_dataset(
            "forces_bias", data=F_bias, maxshape=(None, n_atoms, 3), dtype=np.float64
        )

    def dump_g(self):
        append_to_ds(
            self.data_h5["descriptors/full"], self.ref_cvs[-1], batch_dim=False
        )
        append_to_ds(
            self.data_h5["descriptors/atomic_numbers"],
            self.ref_atomic_numbers[-1],
            batch_dim=False,
        )

    def dump_labels(self):
        E_base = np.array([self.base_results["energy"]])
        E_bias = np.array([self.bias_results["energy"]])
        append_to_ds(self.data_h5["labels/energy"], E_base)
        append_to_ds(self.data_h5["labels/energy_bias"], E_bias)

        append_to_ds(
            self.data_h5["labels/forces"], self.base_results["forces"][None, ...]
        )
        append_to_ds(
            self.data_h5["labels/forces_bias"], self.bias_results["forces"][None, ...]
        )

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
            self._initialize_g_ds()
            self._initialize_nl(atoms)
        if len(self.ref_cvs) == 0:
            self.add_configs([atoms])

        should_update_bias = self._step_counter % self.interval == 0
        if should_update_bias and self.accumulate:
            self.update_bias(atoms)
            self.dump_g()

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
        self.bias_results = {
            k: np.array(v, dtype=np.float64) for k, v in bias_results.items()
        }

        if self._step_counter == 0:
            self._initialize_label_ds()  # would writing else solve the label doubling?
        self.dump_labels()

        self.results = {
            "energy": self.base_results["energy"] + self.bias_results["energy"],
            "forces": self.base_results["forces"] + self.bias_results["forces"],
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


def unbias_trajectory(traj_path, label_path):
    f = h5py.File(label_path, "r", libver="latest")
    E_unbiased = np.array(f["labels/energy"])[1:]
    F_unbiased = np.array(f["labels/forces"])[1:]

    traj = read(traj_path, ":")
    assert F_unbiased.shape[0] == len(traj)

    for ii, atoms in enumerate(traj):
        del atoms.calc
        atoms.calc = SinglePointCalculator(
            atoms, energy=E_unbiased[ii], forces=F_unbiased[ii]
        )

    new_traj_path = Path(traj_path)
    new_traj_path = new_traj_path.with_stem(new_traj_path.stem + "_unbiased")
    write(new_traj_path.as_posix(), traj)
