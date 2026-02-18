from typing import Optional

import jax.numpy as jnp
import numpy as np
from apax.config.train_config import Config
from apax.layers.descriptor import GaussianMomentDescriptor
from apax.layers.descriptor.basis_functions import (
    GaussianBasis,
    RadialFunction,
)
from apax.nn.models import FeatureModel
from apax.utils.jax_md_reduced import partition, space
from functools import partial


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
