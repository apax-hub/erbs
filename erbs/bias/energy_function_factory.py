import jax.numpy as jnp
import numpy as np
from ase import units
from jax import vmap
from jraph import segment_mean
import jax

from erbs.bias.kernel import gaussian


def energy_fn_factory(cv_fn, dim_reduction_fn, Z, g_ref, g_nl, k, a):
    def energy_fn(positions, neighbor):
        g = cv_fn(positions, neighbor)
        g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
        g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
        bias = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

        return jnp.sum(bias)

    return energy_fn
from jax import debug

class OPESExploreFactory:
    def __init__(self, T=300, dE=1.2, a=0.3) -> None:
        self.beta = 1 / (units.kB * T)
        self.std = a
        self.a = None
        self.k = None
        if dE < units.kB * T:
            raise ValueError("dE needs to be larger than 1.0!")
        self.dE = dE
        self.gamma = self.dE * self.beta


    def compute_normalization(self, cluster_models, cluster_idxs, g_ref):
        # z = np.array(Z)
        # elements = np.unique(Z)
        # n_elements = np.max(elements) + 1

        # mc_norm = np.zeros(n_elements) # Z in the paper

        # for element in elements:
        #     g_filtered = g_ref[Z==element]
        #     g_diff = g_filtered[None, :,:] - g_filtered[:, None, :]
        #     s_kk = np.sum((g_diff)**2, axis=2)
        #     G_skk = self.k * np.exp(-s_kk / (self.a * 2.0))
        #     G_skk = np.sum(G_skk)
        #     Zn = G_skk / g_filtered.shape[0]
        #     mc_norm[element] = Zn
        # return jnp.asarray(mc_norm)
        total_n_clusters = 0
        elements = sorted(list(cluster_models.keys()))
        mc_norm = np.zeros(max(elements) + 1) # Z in the paper

        for element in elements:
            current_n_clusters = 0
            elemental_n_data = 0
            elemental_norm = 0.0
            for cluster in range(cluster_models[element].n_clusters):
                current_n_clusters += 1
                cluster_with_offset = cluster + total_n_clusters
                # print(g_ref.shape)
                # print(cluster_idxs)
                # print(cluster_with_offset)
                g_filtered = g_ref[cluster_idxs==cluster_with_offset]

                g_diff = g_filtered[None, :,:] - g_filtered[:, None, :]
                s_kk = np.sum((g_diff)**2, axis=2)
                G_skk = self.k * np.exp(-s_kk / (self.a * 2.0))
                G_skk = np.sum(G_skk)
        
                elemental_norm += G_skk
                elemental_n_data += g_filtered.shape[0]

            total_n_clusters += current_n_clusters
            mc_norm[element] = elemental_norm / elemental_n_data

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

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, Z, neighbor.idx)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, cluster_idxs)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            # prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)
            unnormalized_prob_i = jax.ops.segment_sum(kde_ij, g_nl[0], num_segments=n_atoms)
            prob_i = unnormalized_prob_i / Zn[Z]

            prefactor = (self.gamma - 1.0) / self.beta
            eps = jnp.exp(-(self.dE / prefactor )/(n_atoms))
            
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE
            return total_bias

        return energy_fn


class MetaDCutFactory:
    def __init__(self, k=0.05, a=0.3, E_max=0.043) -> None:
        # self.beta = 1 / (units.kB * T)
        self.a = a
        self.k = k
        self.E_max = E_max

    def create(self, cv_fn, dim_reduction_fn, Z, g_ref, Z_ref, g_nl):
        n_atoms = Z.shape[0]
        cv_dim = g_ref.shape[-1]

        a = self.a ** (cv_dim * 2)
        k = self.k

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            bias_ij = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

            bias_i = jax.ops.segment_sum(bias_ij, g_nl[0], num_segments=n_atoms)

            bias_clipped = jnp.clip(bias_i, a_max=self.E_max)
            bias_cutoff = (self.E_max*n_atoms + A_min - A_curr)/n_atoms * jnp.sin(np.pi/2 * bias_clipped / self.E_max)
            bias = jnp.sum(bias_cutoff)
            bias_scaled_clipped = jnp.clip(bias, a_min=0.0)

            return bias_scaled_clipped

        return energy_fn