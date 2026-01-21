from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from apax.config.train_config import Config
from apax.data.input_pipeline import CachedInMemoryDataset
from apax.layers.descriptor import GaussianMomentDescriptor
from apax.layers.descriptor.basis_functions import (
    GaussianBasis,
    RadialFunction,
)
from apax.nn.models import FeatureModel
from apax.train.checkpoints import restore_parameters
from apax.utils.jax_md_reduced import partition, space
from ase.calculators.calculator import Calculator, all_changes
from matplotlib.path import Path
from tqdm import trange

from erbs.bias.energy_function_factory import OPESExploreFactory
from erbs.bias.state import BiasState
from erbs.dim_reduction.elementwise_pca import DimReduction


def build_feature_neighbor_fns(
    atoms,
    n_basis,
    r_max,
    dr_threshold,
    feature_fn: Optional[callable] = None,
    config: Optional[Config] = None,
    params=None,
    batched=False,
):
    box = np.asarray(atoms.get_cell().lengths(), dtype=jnp.float32)

    if batched:
        displacement_fn = None
        neighbor_fn = None
    else:
        if np.all(box < 1e-6):
            displacement_fn, _ = space.free()
            frac_coords = False
        else:
            displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)
            frac_coords = True
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box,
            r_max,
            dr_threshold,
            fractional_coordinates=frac_coords,
            disable_cell_list=True,
            format=partition.Sparse,
        )

    if config and params:
        n_species = 119  # int(np.max(Z) + 1)
        Builder = config.model.get_builder()
        builder = Builder(config.model.get_dict(), n_species=n_species)

        feature_model = builder.build_ll_feature_model(
            apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
        )
        feature_fn = partial(feature_model.apply, params)
    else:
        descriptor = GaussianMomentDescriptor(
            radial_fn=RadialFunction(
                n_basis,
                basis_fn=GaussianBasis(
                    n_basis=n_basis,
                    r_min=1.5,
                    r_max=r_max,
                ),
                emb_init=None,
            ),
            n_contr=8,
        )
        feature_model = FeatureModel(
            descriptor,
            readout=None,
            should_average=True,
            init_box=box,
            inference_disp_fn=displacement_fn,
        )
        feature_fn = partial(feature_model.apply, {})
    return feature_fn, neighbor_fn


class ERBS(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        base_calc: Calculator,
        dim_reduction_factory: DimReduction,
        energy_fn_factory: OPESExploreFactory,
        feature_fn: Optional[callable] = None,
        model_dir: Optional[Union[Path, list[Path]]] = None,
        n_basis=5,
        r_max=6.0,
        dr_threshold=0.5,
        interval=10_000,
        update_iterations=np.inf,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        # if not isinstance(base_calc, Calculator):
        #     raise ValueError(
        #         "All the calculators should be inherited from"
        #         "the ase's Calculator class"
        #     )
        self.base_calc = base_calc
        self.n_basis = n_basis
        self.model_config = None
        self.params = None
        self.feature_fn = feature_fn
        if model_dir:
            self.model_config, self.params = restore_parameters(model_dir)
        self.r_max = r_max
        self.dr_threshold = dr_threshold
        self.update_iterations = update_iterations

        self.cv_fn = None
        self.dim_reduction_factory = dim_reduction_factory
        self.dim_red_fn = None
        self.energy_fn_factory = energy_fn_factory

        self.energy_fn = None
        self.body_fn = None

        self.auxilliary_cvs = []  # used for dimensionality reduction
        self.ref_cvs = []
        self.bias_state = None
        self.neighbors = None
        self.neighbor_fn = None

        self.interval = interval
        self._step_counter = 0
        self.accumulate = True

        self.bias_results = None

    def _initialize_nl(self, atoms):
        self.cv_fn, self.neighbor_fn = build_feature_neighbor_fns(
            atoms,
            self.n_basis,
            self.r_max,
            self.dr_threshold,
            feature_fn=self.feature_fn,
            config=self.model_config,
            params=self.params,
        )
        self.cv_fn = jax.jit(self.cv_fn)

    def update_with_new_dimred(self, g_new):
        self.ref_cvs.append(g_new)

        reduced_ref_cvs = self.dim_reduction_factory.fit_transform(
            np.array(self.ref_cvs + self.auxilliary_cvs)
        )
        self.dim_red_fn = self.dim_reduction_factory.create_dim_reduction_fn()
        self.dim_red_fn = jax.jit(self.dim_red_fn)

        # create energy fn with new dim_reduction_fn
        self.energy_fn = self.energy_fn_factory.create(
            self.cv_fn,
            self.dim_red_fn,
        )

        threshold = self.energy_fn_factory.compression_threshold

        self.bias_state = BiasState(
            std=self.energy_fn_factory.std,
            g=reduced_ref_cvs,
            compression_threshold=threshold,
        )
        self.bias_state = self.bias_state.initialize()

        if len(reduced_ref_cvs) > 2:
            self.bias_state = self.bias_state.compress()

    def update_with_fixed_dimred(self, g_new):
        self.ref_cvs.append(g_new)
        g_new_red = self.dim_red_fn(g_new)
        if self.bias_state is None:
            raise ValueError("Bias state has not yet been initialized")
        self.bias_state.add_configuration(g_new_red)

    def update_neighbors(self, position, box, is_pbc):
        if self.neighbors is None:
            if is_pbc:
                self.neighbors = self.neighbor_fn.allocate(position, box=box)
            else:
                self.neighbors = self.neighbor_fn.allocate(position)
        else:
            if is_pbc:
                self.neighbors = self.neighbors.update(position, box=box)
            else:
                self.neighbors = self.neighbors.update(position)

    def update_bias(self, atoms):
        position = jnp.array(atoms.positions, dtype=jnp.float64)
        numbers = jnp.array(atoms.numbers, dtype=jnp.int32)

        box = jnp.asarray(atoms.cell.array)

        is_pbc = np.any(atoms.get_cell().lengths() > 1e-6)

        if is_pbc:
            box = box.T
            inv_box = jnp.linalg.inv(box)
            position = space.transform(inv_box, position)

        self.update_neighbors(position, box, is_pbc)

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            if is_pbc:
                self.neighbors = self.neighbor_fn.allocate(position, box=box)
            else:
                self.neighbors = self.neighbor_fn.allocate(position)

        offsets = jnp.zeros((self.neighbors.idx.shape[1], 3))
        g_new = self.cv_fn(position, numbers, self.neighbors.idx, box, offsets)

        should_reinit = self._step_counter < self.update_iterations
        if self.bias_state is None or should_reinit:
            self.update_with_new_dimred(g_new)
        else:
            self.update_with_fixed_dimred(g_new)

        @jax.jit
        def body_fn(positions, neighbor, box, bias_state):
            if np.any(atoms.get_cell().lengths() > 1e-6):
                box = box.T
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)
                neighbor = neighbor.update(positions, box=box)
            else:
                neighbor = neighbor.update(positions)

            offsets = jnp.full([neighbor.idx.shape[1], 3], 0)

            ef_function = jax.value_and_grad(self.energy_fn)
            energy, neg_forces = ef_function(
                positions, numbers, neighbor, box, offsets, bias_state
            )
            forces = -neg_forces
            results = {"energy": energy, "forces": forces}
            return results, neighbor

        self.body_fn = body_fn

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.base_calc.calculate(atoms, properties, system_changes)
        self.results = self.base_calc.results

        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)

        if self._step_counter == 0:
            self._initialize_nl(atoms)

        should_update_bias = self._step_counter % self.interval == 0
        if should_update_bias and self.accumulate:
            self.update_bias(atoms)

        bias_results, self.neighbors = self.body_fn(
            positions, self.neighbors, box, self.bias_state
        )

        if self.neighbors.did_buffer_overflow:
            print("neighbor list overflowed, reallocating.")
            self.neighbors = self.neighbor_fn.allocate(positions, box=box)
            bias_results, self.neighbors = self.body_fn(
                positions, self.neighbors, box, self.bias_state
            )

        self.bias_results = {
            k: np.array(v, dtype=np.float64) for k, v in bias_results.items()
        }

        self.results["energy"] = self.results["energy"] + self.bias_results["energy"]
        self.results["forces"] = self.results["forces"] + self.bias_results["forces"]
        self.results["energy_bias"] = self.bias_results["energy"]
        self.results["forces_bias"] = self.bias_results["forces"]

        self._step_counter += 1

    def compute_cvs(self, atoms_list, batch_size=4):
        dataset = CachedInMemoryDataset(
            atoms_list,
            self.r_max,
            batch_size,
            n_epochs=1,
            ignore_labels=True,
        )

        n_data = dataset.n_data
        ds = dataset.batch()

        self.cv_fn, _ = build_feature_neighbor_fns(
            atoms_list[0],
            self.n_basis,
            self.r_max,
            dr_threshold=self.dr_threshold,
            batched=True,
        )

        def calc_descriptor(positions, Z, neighbors, box, offsets):
            g = self.cv_fn(positions, Z, neighbors, box, offsets)
            return g

        calc_descriptor = jax.vmap(calc_descriptor, in_axes=(0, 0, 0, 0, 0))
        calc_descriptor = jax.jit(calc_descriptor)

        descriptors = []

        pbar = trange(n_data, desc="Evaluating data", ncols=100, leave=False)
        for i, inputs in enumerate(ds):
            g = calc_descriptor(
                inputs["positions"],
                inputs["numbers"],
                inputs["idx"],
                inputs["box"],
                inputs["offsets"],
            )

            num_strucutres_in_batch = g.shape[0]
            for j in range(num_strucutres_in_batch):
                g_cpu = np.asarray(g[j])
                descriptors.append(g_cpu)

            pbar.update(batch_size)
        pbar.close()
        dataset.cleanup()

        return descriptors

    def add_configs(self, atoms_list, batch_size=4, for_dimred_only=True):
        descriptors = self.compute_cvs(atoms_list, batch_size)

        if for_dimred_only:
            self.auxilliary_cvs.extend(descriptors)
        else:
            self.ref_cvs.extend(descriptors)

    def add_descriptors(self, path, for_dimred_only=True):
        data = np.load(path)
        descriptors = data["g"]

        descriptors = list(descriptors)
        if for_dimred_only:
            self.auxilliary_cvs.extend(descriptors)
        else:
            self.ref_cvs.extend(descriptors)

    def save_descriptors(self, path):
        data = {"g": np.array(self.ref_cvs)}
        if len(self.auxilliary_cvs) > 0:
            data["g_aux"] = np.array(self.auxilliary_cvs)
        np.savez(path, **data)
