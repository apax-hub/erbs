from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import h5py

from erbs.cv.cv_nl import compute_cv_nl

def append_to_ds(ds, new_data):
    shape = ds.shape[0] + new_data.shape[0]
    ds.resize(shape, axis=0)
    ds[-new_data.shape[0]:,...] = new_data


class GKernelMTD(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calc,
        cv_fn,
        dim_reduction_factory,
        energy_fn_factory,
        neighbor_fn,
        k,
        a,
        interval=100,
        log_file="labels.hdf5",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        if not isinstance(base_calc, Calculator):
            raise ValueError(
                "All the calculators should be inherited from" \
                "the ase's Calculator class"
            )
        self.base_calc = base_calc
        self.log_file = log_file

        self.k = k
        self.a = a

        self.cv_fn = cv_fn
        self.dim_reduction_factory = dim_reduction_factory
        self.energy_fn_factory = energy_fn_factory
        self.neighbor_fn = neighbor_fn

        self.energy_and_force_fn = None

        self.ref_cvs = []
        # self.reduced_ref_cvs = None
        # ^ maybe introduce if we allow for reduction function update interval
        self.ref_atomic_numbers = []
        self.neighbors = None

        self.interval = interval
        self._step_counter = 0
        self.accumulate = True
        # self.data_h5 = None
        self.data_h5 = h5py.File(self.log_file, "w", libver='latest')
        self.initialized = False

    def _initialize_g_ds(self):
        g_grp = self.data_h5.create_group("descriptors")
        cv_shape = self.ref_cvs[0].shape#[1] + list(self.ref_cvs[0].shape)
        g_grp.create_dataset("full", data=self.ref_cvs[0][::-1], maxshape=(None, cv_shape[1]), dtype=np.float32)
        g_grp.create_dataset("atomic_numbers", data=self.ref_atomic_numbers[0][::-1], maxshape=(None, ), dtype=np.int32)


    def _initialize_label_ds(self):

        label_grp = self.data_h5.create_group("labels")
        E_base = self.energy_buffer#np.array([self.base_results["energy"]])
        E_bias = self.energy_bias_buffer#np.array([self.bias_results["energy"]])
        
        label_grp.create_dataset("energy", data=E_base, maxshape=(None,), dtype=np.float64)
        label_grp.create_dataset("energy_bias", data=E_bias, maxshape=(None,), dtype=np.float64)

        n_atoms = self.base_results["forces"].shape[0]
        label_grp.create_dataset("forces", data=self.forces_buffer, maxshape=(None,n_atoms,3), dtype=np.float64)
        label_grp.create_dataset("forces_bias", data=self.forces_bias_buffer, maxshape=(None,n_atoms,3), dtype=np.float64)


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

        energy_fn = self.energy_fn_factory(
            self.cv_fn,
            self.dim_reduction_factory.create_dim_reduction_fn(),
            numbers,
            reduced_ref_cvs,
            g_neighbors,
            self.k,
            self.a,
        )

        @jax.jit
        def body_fn(positions, neighbor):
            neighbor = neighbor.update(positions)
            energy, neg_forces = jax.value_and_grad(energy_fn)(positions, neighbor)
            forces = -neg_forces

            return {"energy": energy, "forces": forces}, neighbor

        self.energy_and_force_fn = body_fn


    def dump_data(self):
        append_to_ds(self.data_h5["descriptors/full"], self.ref_cvs[-1])
        append_to_ds(self.data_h5["descriptors/atomic_numbers"], self.ref_atomic_numbers[-1])

        append_to_ds(self.data_h5["labels/energy"], self.energy_buffer)
        append_to_ds(self.data_h5["labels/forces"], self.forces_buffer)

        append_to_ds(self.data_h5["labels/energy_bias"], self.energy_bias_buffer)
        append_to_ds(self.data_h5["labels/forces_bias"], self.forces_bias_buffer)

    # def dump_g(self):
    #     append_to_ds(self.data_h5["descriptors/full"], self.ref_cvs[-1])
    #     append_to_ds(self.data_h5["descriptors/atomic_numbers"], self.ref_atomic_numbers[-1])

    # def dump_labels(self):
    #     append_to_ds(self.data_h5["labels/energy"], self.energy_buffer)
    #     append_to_ds(self.data_h5["labels/forces"], self.forces_buffer)

    #     append_to_ds(self.data_h5["labels/energy_bias"], self.energy_bias_buffer)
    #     append_to_ds(self.data_h5["labels/forces_bias"], self.forces_bias_buffer)
        
    def reset_buffer(self, n_atoms):
        self.energy_buffer = np.zeros(self.interval)
        self.forces_buffer = np.zeros((self.interval, n_atoms, 3))
        self.energy_bias_buffer = np.zeros(self.interval)
        self.forces_bias_buffer = np.zeros((self.interval, n_atoms, 3))

    def write_to_buffer(self):
        i = self._step_counter % self.interval
        print(self._step_counter, i)
        self.energy_buffer[i] = self.base_results["energy"]
        self.forces_buffer[i] = self.base_results["forces"]
        self.energy_bias_buffer[i] = self.bias_results["energy"]
        self.forces_bias_buffer[i] = self.bias_results["forces"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.base_calc.calculate(atoms, properties, system_changes)
        self.base_results = self.base_calc.results

        should_update_bias = self._step_counter % self.interval == 0
        if should_update_bias and self.accumulate:
            self.update_bias(atoms)

        if self._step_counter == 0:
            self._initialize_g_ds()
            self.reset_buffer(len(atoms))

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
        self.bias_results = {k: np.array(v, dtype=np.float64) for k,v in bias_results.items()}
        
        # print(self.base_results["energy"], self.bias_results["energy"])
        print(self.base_results["energy"])
        if should_update_bias:
            self.reset_buffer(len(atoms))
        self.write_to_buffer()

        print(self.energy_buffer)
        should_dump = self._step_counter % self.interval == 1
        if should_dump:
            # print(self._step_counter)
            if self._step_counter >= self.interval and not self.initialized:
                self._initialize_label_ds()
                self.initialized = True
            elif self.initialized:
                self.dump_data()

        self.results = {
            "energy": self.base_results["energy"], #+ self.bias_results["energy"],
            "forces": self.base_results["forces"], #+ self.bias_results["forces"],
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

            if not self.neighbors or self.neighbors.did_buffer_overflow:
                self.neighbors = self.neighbor_fn.allocate(positions)
            g = calc_g(positions, self.neighbors)
            self.ref_cvs.append(np.asarray(g))
            self.ref_atomic_numbers.append(np.asarray(numbers))
