import numpy as np
from sklearn.decomposition import PCA

class ElementwisePCA:
    def __init__(self, n_components=2) -> None:
        self.n_components = n_components

        self.mu = None
        self.sigmaT = None

    def fit_transform(self, g, Z):
        elements = np.unique(Z)
        # TODO reference g and Z should probably be sorted separably
        # since this method sorts them
        # and subsequent functions can make use of sorted lists

        new_gs = []
        new_zs = []
        mu = np.zeros((np.max(Z), g.shape[1]))
        sigmaT = np.zeros((np.max(Z), g.shape[1], self.n_components))

        for element in elements:
            g_filtered = g[Z==element]
            pca = PCA(n_components=self.n_components)
            g_pca = pca.fit_transform(g_filtered)
            
            new_gs.append(g_pca)
            new_zs.append(Z[Z==element])

            mu[element] = pca.mean_
            sigmaT[element] = pca.components_.T

        new_gs = np.concatenate(new_gs, axis=0)
        new_zs = np.concatenate(new_zs, axis=0)
        self.me, self.sigmaT = mu, sigmaT
        return new_gs, new_zs

    def create_dim_reduction_fn(self):

        def dim_reduction_fn(g_i, Z_i):
            g_reduced_i = ((g_i - self.mu[Z_i]) @ self.sigmaT[Z_i])
            return g_reduced_i

        return dim_reduction_fn