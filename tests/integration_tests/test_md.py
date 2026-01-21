import pytest
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

# Adjust these imports to match your actual package structure
from erbs.bias import ERBS
from erbs.dim_reduction import GlobalPCA
from erbs.bias import OPESExploreFactory

def test_erbs_md_integration():
    """
    Integration test for ERBS calculator wrapper.
    Verifies that the calculator can be instantiated and drives 
    a stable MD simulation for a few steps.
    """
    # 1. Setup Test System (Water molecule)
    # Using a simple molecule to keep the test fast
    atoms = molecule("H2O")
    atoms.center(vacuum=4.0)
    
    # 2. Setup Base Calculator
    # EMT is a fast, built-in potential in ASE (good for testing)
    base_calc = EMT()
    
    # 3. Define ERBS Parameters
    temperature = 300.0
    barrier_factor = 10.0
    bias_interval = 1
    
    # PCA Setup
    # Assuming 'n_components' is the first arg based on your snippet
    pca = GlobalPCA(n_components=2, skip_first_n_components=None)

    # Bias Factory Setup
    dE = units.kB * temperature * barrier_factor
    # 'a' usually corresponds to the bandwidth (sigma) of the kernels
    energy_fn_factory = OPESExploreFactory(T=temperature, dE=dE, a=1.0)

    # 4. Instantiate ERBS Calculator
    # Using the exact signature you provided
    calc = ERBS(
        base_calc=base_calc,
        dim_reduction_factory=pca,
        energy_fn_factory=energy_fn_factory,
        n_basis=4,         # Small basis for testing
        r_min=0.5,          # Typical bond range start
        r_max=3.0,          # Cutoff
        dr_threshold=0.5,   # Neighborlist skin
        interval=bias_interval,
    )
    
    atoms.calc = calc

    # 5. Run MD Loop
    # Initialize momenta
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    
    # Use standard Velocity Verlet
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    
    initial_energy = atoms.get_potential_energy()
    
    # Run a few steps to ensure dynamics propagate
    try:
        dyn.run(steps=10)
    except Exception as e:
        pytest.fail(f"MD simulation failed with error: {e}")

    # 6. Assertions
    final_energy = atoms.get_potential_energy()
    
    # Check that energy is finite
    assert not np.isnan(final_energy)
    assert not np.isinf(final_energy)
    
    # Check that the system actually moved (energy changed)
    # In a biased simulation, energy should fluctuate
    assert final_energy != initial_energy

    # Optional: Check if ERBS specific results are stored
    # This depends on your implementation, but usually wrappers store 
    # the bias energy separately
    if 'bias_energy' in calc.results:
        assert calc.results['bias_energy'] >= 0
