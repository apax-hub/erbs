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

    def compute_normalization(self, Z, g_ref):
        # z = np.array(Z)
        elements = np.unique(Z)
        n_elements = np.max(elements) + 1

        mc_norm = np.zeros(n_elements) # Z in the paper
        # print()

        for element in elements:
            g_filtered = g_ref[Z==element]
            g_diff = g_filtered[None, :,:] - g_filtered[:, None, :]
            # print(g_diff.shape)
            s_kk = np.sum((g_diff)**2, axis=2)
            G_skk = self.k * np.exp(-s_kk / (self.a * 2.0))
            # print(G_skk.shape)
            # print(G_skk)
            G_skk = np.sum(G_skk)
            # print(G_skk)
            Zn = G_skk / g_filtered.shape[0]
            # print(G_skk, Zn, g_filtered.shape[0])
            mc_norm[element] = Zn
        # print(mc_norm)
        # quit()
        return jnp.asarray(mc_norm)

    def create(self, cv_fn, dim_reduction_fn, Z, g_ref, Z_ref, g_nl):
        n_atoms = Z.shape[0]
        cv_dim = g_ref.shape[-1]

        self.a = self.std ** (cv_dim * 2)
        self.k = 1 / (np.sqrt(2.0 * np.pi * self.std**2) **cv_dim )
        # print(self.a)
        # print(self.k)
        # quit()

        mc_norm = self.compute_normalization(Z_ref, g_ref)
        # print(self.dE)
        # print((self.dE * self.beta / (self.gamma - 1.0))/n_atoms * (self.gamma - 1.0) / self.beta * n_atoms)
        # quit()
        # prefactor = (self.gamma - 1.0) / self.beta
        # print(prefactor)
        # print(jnp.exp(-(self.dE * self.beta / (self.gamma - 1.0))/(n_atoms)), jnp.exp(-(self.dE * self.beta / (self.gamma - 1.0))))
        # quit()

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            kde_ij = vmap(gaussian, (0, None, None), 0)(g_diff, self.k, self.a)

            prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)
            # unnormalized_prob_i = jax.ops.segment_sum(kde_ij, g_nl[0], num_segments=n_atoms)
            # prob_i = unnormalized_prob_i / mc_norm[Z]

            # prefactor = 1/ self.beta
            prefactor = (self.gamma - 1.0) / self.beta
            eps = jnp.exp(-(self.dE / prefactor )/(n_atoms))
            
            bias_i = prefactor * jnp.log(prob_i + eps)
            total_bias = jnp.sum(bias_i) + self.dE

            # debug.print("")
            # debug.print("g_nl[0] {x}", x=g_nl[0])
            # debug.print("unnormalized_prob_i {x}", x=unnormalized_prob_i)
            # debug.print("prob_i {x}", x=prob_i)
            # debug.print("mc_norm[Z] {x}", x=mc_norm[Z])
            # debug.print("eps {x}", x=eps)
            # debug.print("bias_i {x}", x=bias_i)
            # debug.print("total_bias {x}", x=total_bias)

            return total_bias # bias_i

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
        # k = self.k / (np.sqrt(2.0 * np.pi * self.a**2) **cv_dim )
        k = self.k

        def energy_fn(positions, neighbor, A_curr, A_min):
            g = cv_fn(positions, neighbor)
            g_reduced = vmap(dim_reduction_fn, 0, 0)(g, Z)
            g_diff = g_reduced[g_nl[0]] - g_ref[g_nl[1]]
            bias_ij = vmap(gaussian, (0, None, None), 0)(g_diff, k, a)

            # prob_i = segment_mean(kde_ij, g_nl[0], num_segments=n_atoms)
            bias_i = jax.ops.segment_sum(bias_ij, g_nl[0], num_segments=n_atoms)
            # debug.print("bias_ij {x}", x=bias_ij)
            
            # bias = jnp.sum(bias_i)

            # debug.print("bias {x}", x=bias)
            # debug.print("bias_clip {x}", x=bias_clipped)
            # debug.print("self.E_max {x}", x=self.E_max)
            # debug.print("A_min {x}", x=A_min)
            # debug.print("A_curr {x}", x=A_curr)
            # debug.print("A_curr {x}", x=self.E_max + A_min - A_curr)

            # bias_clipped = jnp.clip(bias, a_max=self.E_max)
            # bias_limited = (self.E_max + A_min - A_curr) * jnp.sin(np.pi/2 * bias_clipped / self.E_max)# should be smooth at 0 too e.g via sigmoid
            # bias_clipped = jnp.clip(bias_limited, a_min=0.0)

            # bias_clipped = jnp.clip(bias_i, a_max=self.E_max)
            # # bias_limited = self.E_max* jnp.sin(np.pi/2 * bias_clipped / self.E_max)# should this be smooth at 0 too e.g via sigmoid
            # bias_cutoff = jnp.sin(np.pi/2 * bias_clipped / self.E_max)
            # bias = jnp.sum(bias_cutoff)
            # bias_scaled = (self.E_max * n_atoms + A_min - A_curr) * bias
            # bias_scaled_clipped = jnp.clip(bias_scaled, a_min=0.0)


            bias_clipped = jnp.clip(bias_i, a_max=self.E_max)
            # bias_limited = self.E_max* jnp.sin(np.pi/2 * bias_clipped / self.E_max)# should this be smooth at 0 too e.g via sigmoid
            bias_cutoff = (self.E_max*n_atoms + A_min - A_curr)/n_atoms * jnp.sin(np.pi/2 * bias_clipped / self.E_max)
            bias = jnp.sum(bias_cutoff)
            # bias_scaled = (self.E_max * n_atoms + A_min - A_curr) * bias
            bias_scaled_clipped = jnp.clip(bias, a_min=0.0)

            # debug.print("")
            # debug.print("bias_i {x}", x=bias_i)
            # debug.print("bias_clipped {x}", x=bias_clipped)
            # debug.print("bias_cutoff{x}", x=bias_cutoff)
            # debug.print("bias {x}", x=bias)
            # debug.print("(self.E_max * n_atoms + A_min - A_curr) {x}", x=(self.E_max * n_atoms + A_min - A_curr))
            # debug.print("(self.E_max * n_atoms ) {x}", x=(self.E_max * n_atoms ))
            # debug.print("( A_min - A_curr) {x}", x=( A_min - A_curr))
            # # bias_limited = jnp.where(bias_limited < self.E_max, bias_limited, self.E_max)
            # # debug.print("bias_scaled {x}", x=bias_scaled)
            # debug.print("bias_scaled_clipped {x}", x=bias_scaled_clipped)
           

            return bias_scaled_clipped #jnp.sum(bias_limited)

        return energy_fn