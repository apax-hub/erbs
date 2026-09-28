from typing import Protocol, Callable, Optional, Union
from pathlib import Path
from functools import partial
import jax.numpy as jnp
import numpy as np

class FeatureBuilder(Protocol):
    def __call__(self, displacement_fn: Callable, box: jnp.ndarray) -> Callable:
        """
        Builds the feature function.

        Parameters
        ----------
        displacement_fn : Callable
            JAX-MD displacement function.
        box : jnp.ndarray
            Simulation box.

        Returns
        -------
        Callable
            A feature function with signature (positions, numbers, idx, box, offsets) -> descriptor.
        """
        ...

class ApaxBuilder:
    def __init__(self, model_dir: Union[Path, str, list[Path]]):
        from apax.train.checkpoints import restore_parameters
        self.config, self.params = restore_parameters(model_dir)

    def __call__(self, displacement_fn: Optional[Callable], box: jnp.ndarray) -> Callable:
        n_species = 119 # Default for apax
        Builder = self.config.model.get_builder()
        builder = Builder(self.config.model.get_dict(), n_species=n_species)

        feature_model = builder.build_ll_feature_model(
            apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
        )
        return partial(feature_model.apply, self.params)

class DescriptorBuilder:
    def __init__(self, n_basis: int = 5, r_max: float = 6.0):
        self.n_basis = n_basis
        self.r_max = r_max

    def __call__(self, displacement_fn: Optional[Callable], box: jnp.ndarray) -> Callable:
        from apax.layers.descriptor import GaussianMomentDescriptor
        from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
        from apax.nn.models import FeatureModel

        descriptor = GaussianMomentDescriptor(
            radial_fn=RadialFunction(
                self.n_basis,
                basis_fn=GaussianBasis(
                    n_basis=self.n_basis,
                    r_min=1.5,
                    r_max=self.r_max,
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
        return partial(feature_model.apply, {})

class CustomBuilder:
    def __init__(self, feature_fn_factory: Callable[[Optional[Callable], jnp.ndarray], Callable]):
        self.feature_fn_factory = feature_fn_factory

    def __call__(self, displacement_fn: Optional[Callable], box: jnp.ndarray) -> Callable:
        return self.feature_fn_factory(displacement_fn, box)
