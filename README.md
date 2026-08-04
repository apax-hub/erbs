# ERBS - Enhanced Representation Based Sampling

**ERBS** is a JAX-based library for the efficient generation of training data for Machine-Learned Interatomic Potentials (MLIPs).

It implements the Enhanced Representation Based Sampling strategy, which identifies slow, physically relevant collective variables directly from the variance of atomic descriptors and accelerates sampling along these modes using bias potentials.
The library itself ships with a simplified version of the Gaussian Moment descriptor, but is descriptor agostic and can be easily interfaced with others.



## Features

- Automated CV Selection: Automatically identifies slow degrees of freedom via PCA on the descriptor space.
- OPES-Explore: Uses the OPES-Explore formalism to rapidly explore the free energy surface with bounded bias potentials.
- Model Agnostic: Can be used with any atomistic descriptor that matches the required function signature (see Documentation).


## Installation

ERBS is available on PyPI:

```bash
pip install erbs
```

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/apax_hub/erbs.git
```


## Usage

ERBS integrates with ASE to run enhanced sampling simulations.
A typical workflow involves the typical ASE setup, selecting a descriptor builder and wrapping the plain energy/forces calculator with ERBS.

```python
from erbs.interfaces.ase import ERBS
from erbs.biases import OPESExploreFactory
from erbs.dim_reductions import GlobalPCA
from erbs.descriptors.factory import DescriptorBuilder

dim_reduction = GlobalPCA(n_components=pca_components)

dE = units.kB * temperature * barrier_factor
energy_fn_factory = OPESExploreFactory(
    T=temperature, dE=dE, a=band_width
)

# Use the built-in Gaussian Moment descriptor
feature_builder = DescriptorBuilder(n_basis=4, r_max=6.0)

base_calc = ... # Any ASE Calculator
calc = ERBS(
    base_calc=base_calc,
    dim_reduction_factory=dim_reduction,
    energy_fn_factory=energy_fn_factory,
    feature_builder=feature_builder,
    interval=2000,
)
```

Authors

    Moritz René Schäfer

Under the supervision of Johannes Kästner.


## Contributing

We are happy to receive your issues and pull requests!
If you want to add a new dimensionality reduction, energy function or descriptor, please open an issue to discuss the implementation details.



## Acknowledgements

The creation of ERBS was supported by the DFG under Germany's Excellence Strategy - EXC 2075 - 390740016 and the Stuttgart Center for Simulation Science (SimTech).


References

If you use ERBS in your research, please cite the following paper:

TBD
