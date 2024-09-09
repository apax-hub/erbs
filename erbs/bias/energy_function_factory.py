import jax.numpy as jnp
import numpy as np
from ase import units
from jax import vmap
from erbs.ops import segment_mean
import jax

from erbs.bias.kernel import gaussian


class MetaDFactory:
    def __init__(self, k=1.2, a=0.3) -> None:
        self.std = a
        self.k = k

    def create(self, cv_fn, dim_reduction_fn, cluster_models, Z, cluster_idxs, g_ref, cluster_idxs_ref, g_nl):
        Z = jnp.asarray(Z)
        cv_dim = g_ref.shape[-1]

        self.a = self.std ** (cv_dim * 2)

        cluster_idxs = jnp.asarray(cluster_idxs)

        def energy_fn(positions, neighbor):
            g = cv_fn(positions, Z, neighbor.idx)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, cluster_idxs)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            total_bias = jnp.sum(kde_ij)
            return total_bias

        return energy_fn


def chunked_sum_of_kernels(X, k, a, chunk_size=50):

    n_chunks = X.shape[0] // chunk_size
    if X.shape[0] % chunk_size > 0:
        n_chunks += 1

    G_skk = 0
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = i * chunk_size + chunk_size
        for j in range(n_chunks):
            start_j = j * chunk_size
            end_j = j * chunk_size + chunk_size

            gdiff = X[None, start_i:end_i,:] - X[start_j:end_j, None, :]
            s_kk = np.sum(gdiff, axis=2)**2
            G_skk_chunk = k * np.exp(-s_kk / (a * 2.0))
            G_skk += np.sum(G_skk_chunk)

    return G_skk


class OPESExploreFactory:
    def __init__(self, T=300, dE=1.2, a=0.3, use_mc_norm=True) -> None:
        self.beta = 1 / (units.kB * T)
        self.std = a
        self.a = None
        self.k = None
        if dE < units.kB * T:
            raise ValueError("dE needs to be larger than 1.0!")
        self.dE = dE
        self.gamma = self.dE * self.beta
        self.use_mc_norm = use_mc_norm


    def compute_normalization(self, cluster_models, cluster_idxs, g_ref):

        total_n_clusters = 0
        elements = sorted(list(cluster_models.keys()))
        mc_norm = np.zeros(np.max(cluster_idxs) + 1) # Z in the paper

        for element in elements:
            current_n_clusters = 0
            for cluster in range(cluster_models[element].n_clusters):
                current_n_clusters += 1
                cluster_with_offset = cluster + total_n_clusters
                g_filtered = g_ref[cluster_idxs==cluster_with_offset]

                G_skk = chunked_sum_of_kernels(g_filtered, self.k, self.a)

                mc_norm[cluster_with_offset] = G_skk / g_filtered.shape[0]

            total_n_clusters += current_n_clusters
        return mc_norm

    def create(self, cv_fn, dim_reduction_fn, cluster_models, Z, cluster_idxs, g_ref, cluster_idxs_ref, g_nl):
        n_atoms = Z.shape[0]
        Z = jnp.asarray(Z)
        cv_dim = g_ref.shape[-1]

        self.a = self.std ** (cv_dim * 2)
        self.k = 1 / (np.sqrt(2.0 * np.pi * self.std**2) **cv_dim )

        Zn = self.compute_normalization(cluster_models, cluster_idxs_ref, g_ref)
        Zn = jnp.asarray(Zn)
        cluster_idxs = jnp.asarray(cluster_idxs)

        def energy_fn(positions, Z, neighbor, box, offsets):
            g = cv_fn(positions, Z, neighbor.idx, box, offsets)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, cluster_idxs)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            if self.use_mc_norm:
                unnormalized_prob_i = jax.ops.segment_sum(kde_ij, g_nl[0], num_segments=n_atoms)
                prob_i = unnormalized_prob_i / Zn[cluster_idxs]
            else:
                prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)

            prefactor = (self.gamma - 1.0) / self.beta

            eps = jnp.exp(-(self.dE / prefactor )/(n_atoms))
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE

            return total_bias

        return energy_fn
