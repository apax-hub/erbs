import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


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

        g = np.concatenate(g, axis=0)
        Z = np.concatenate(Z, axis=0)
        g = np.reshape(g, (-1, g.shape[-1]))
        Z = np.reshape(Z, (-1,))

        n_elements = np.max(Z) + 1

        new_gs = []
        new_zs = []
        mu = np.zeros((n_elements, g.shape[1]))
        sigmaT = np.zeros((n_elements, g.shape[1], self.n_components))

        for element in elements:
            g_filtered = g[Z == element]
            pca = PCA(n_components=self.n_components)
            g_pca = pca.fit_transform(g_filtered)

            new_gs.append(g_pca)
            new_zs.append(Z[Z == element])

            mu[element] = pca.mean_
            sigmaT[element] = pca.components_.T

        new_gs = np.concatenate(new_gs, axis=0)
        new_zs = np.concatenate(new_zs, axis=0)
        self.mu, self.sigmaT = jnp.array(mu), jnp.array(sigmaT)
        return new_gs, new_zs

    def create_dim_reduction_fn(self):
        def dim_reduction_fn(g_i, Z_i):
            g_reduced_i = (g_i - self.mu[Z_i]) @ self.sigmaT[Z_i]
            return g_reduced_i

        return dim_reduction_fn


def kmeans_silhouette_score(estimator, X):
    score = silhouette_score(X, estimator.labels_, metric = 'euclidean')
    return score


class KMeansFallback:

    def __init__(self) -> None:
        self.labels_ = None
        self.n_clusters = 1

    def fit(self, X):
        self.labels_ = np.zeros((X.shape[0]), dtype=np.int32)
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=np.int32)


class ElementwiseLocalPCA:
    def __init__(self, n_components=2, kmax=10) -> None:
        self.n_components = n_components
        self.kmax = kmax

        self.mu = None
        self.sigmaT = None

        self.clusters_per_element = [None] * 119
        self.cluster_models = {}
        self.cluster_offset_per_element = np.zeros(119, dtype=np.int32)

    def determine_cluster_range(self, element, n_samples):
        current_num_clusters = self.clusters_per_element[element]
        if current_num_clusters is None:
            kmin=2
            kmax = min(n_samples -1, self.kmax)
        else:
            kmin = max(2, current_num_clusters-1)
            kmax = min(n_samples -1, current_num_clusters+1)
        return range(kmin, kmax)
    
    def fit_clustering(self, element, X):
        n_samples = X.shape[0]

        cluster_range = self.determine_cluster_range(element, n_samples)
        sil = []
        kmean_models = []
        if n_samples <= 2:
            fallback = KMeansFallback()
            fallback.fit(X)
            self.clusters_per_element[element] = None
            return fallback

        for k in cluster_range:
            # print(k, X.shape[0])
            kmeans = KMeans(n_clusters = k, n_init="auto", init="k-means++").fit(X)
            labels = kmeans.labels_
            sil.append(silhouette_score(X, labels, metric = 'euclidean'))
            kmean_models.append(kmeans)

        best_model_idx = np.argmax(sil)
        best_model = kmean_models[best_model_idx]
        return best_model

    def fit_transform(self, g, Z):
        elements = np.unique(Z)
        # since this method sorts them
        # and subsequent functions can make use of sorted lists

        g = np.concatenate(g, axis=0)
        Z = np.concatenate(Z, axis=0)
        g = np.reshape(g, (-1, g.shape[-1]))
        Z = np.reshape(Z, (-1,))

        # n_elements = np.max(Z) + 1

        new_gs = []
        cluster_idxs = []
        mu = []
        sigmaT = []
        total_n_clusters = 0
        # self.cluster_element_mapping = {}#{i: [] for i in range(119)}
        

        for element in elements:
            g_z = g[Z == element]
            
            cluster_model = self.fit_clustering(element, g_z)
            self.cluster_models[element] = cluster_model
            # self.cluster_element_mapping[element]
            labels = cluster_model.labels_
            n_clusters = cluster_model.n_clusters
            for cluster_idx in range(n_clusters):
                # self.cluster_element_mapping[element].append
                g_cluster = g_z[labels==cluster_idx]
                idxs = labels[labels==cluster_idx] + total_n_clusters
                n_samples = g_cluster.shape[0]
                if n_samples < self.n_components:
                    mean = np.mean(g_cluster, axis=0)
                    comps = np.random.normal(size=(g_cluster.shape[1], self.n_components))

                    g_pca = (g_cluster - mean) @ comps

                    mu.append(mean)
                    sigmaT.append(comps)
                    new_gs.append(g_pca)
                else:
                    pca = PCA(n_components=self.n_components)
                    g_pca = pca.fit_transform(g_cluster)


                    mu.append(pca.mean_)
                    sigmaT.append(pca.components_.T)
                    new_gs.append(g_pca)

                cluster_idxs.append(idxs)

            self.cluster_offset_per_element[element] = total_n_clusters
            total_n_clusters += n_clusters

        new_gs = np.concatenate(new_gs, axis=0)
        cluster_idxs = np.concatenate(cluster_idxs, axis=0)
        self.mu, self.sigmaT = jnp.array(mu), jnp.array(sigmaT)
        # print("TOTAL CLUSTER", total_n_clusters)
        return new_gs, cluster_idxs
    
    def apply_clustering(self, g, Z):
        
        g = np.asarray(g)
        Z = np.asarray(Z)

        cluster_idxs = []
        for i, z in enumerate(Z):
            if self.cluster_models[z] is None:
                raise ValueError(f"No clustering found for element {z}")
            gi = g[i][None,:]
            cluster_idx = self.cluster_models[z].predict(gi) 
            cluster_idxs.append(cluster_idx + self.cluster_offset_per_element[z])

        cluster_idxs = jnp.array(cluster_idxs, dtype=jnp.int32)
        cluster_idxs = np.reshape(cluster_idxs, (-1,))
        return cluster_idxs

    def create_dim_reduction_fn(self):

        def dim_reduction_fn(g_i, cluster_idx):
            g_reduced_i = (g_i - self.mu[cluster_idx]) @ self.sigmaT[cluster_idx]
            return g_reduced_i

        return dim_reduction_fn