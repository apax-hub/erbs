import numpy as np
from ase.neighborlist import build_neighbor_list

def repartition_hydrogen_mass(atoms, h_mass_scale=3.0):
    atoms = atoms.copy()

    h_mass_scale = 3

    nl = build_neighbor_list(atoms, self_interaction=False)
    cm = nl.get_connectivity_matrix(sparse=False)
    z = atoms.get_atomic_numbers()
    H_idxs = (z==1).nonzero()[0]

    H_connectivity = cm.T[H_idxs]
    num_hs = np.sum(H_connectivity, axis=0)

    m = atoms.get_masses()
    m_scaled = m.copy()
    m_scaled[H_idxs] *= h_mass_scale

    mask = np.ones(len(atoms), dtype=np.int32)
    mask[H_idxs] = 0
    mask = np.nonzero(mask)[0]
    m_scaled[mask] -= 1.008 * num_hs[mask] * (h_mass_scale-1)

    atoms.set_masses(m_scaled)
    return atoms