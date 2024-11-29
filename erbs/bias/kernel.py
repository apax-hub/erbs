import jax.numpy as jnp
import numpy as np

def gaussian(g_diff, k, a):
    x = jnp.sum((g_diff) ** 2)
    U = k * jnp.exp(-x / (a * 2.0))
    return U

def diag_gaussian(gdiff, k, cov):
    x = jnp.dot(gdiff.T, gdiff / cov)
    U = k * jnp.exp(-x/2)
    return U


def chunked_sum_of_kernels(X, k, cov, chunk_size=200):
    # TODO use diagonal gaussian

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
            cov_chunk = cov[start_i:end_i,None, :]
            k_chunk = k[start_i:end_i,:]
            # TODO CHECK CHUNKING FOR COV for chunksize = 1

            x = np.einsum("bcj, bcj -> bc" , gdiff, gdiff / cov_chunk)
            G_skk_chunk = k_chunk * np.exp(- x / 2.0)
            G_skk += np.sum(G_skk_chunk)

    return G_skk


def global_mc_normalisation(g_ref, height, cov):

    G_skk = chunked_sum_of_kernels(g_ref, height, cov)
    mc_norm = G_skk / g_ref.shape[0]

    return mc_norm



def mc_normalisation(cluster_models, cluster_idxs, g_ref, height, var):

    total_n_clusters = 0
    elements = sorted(list(cluster_models.keys()))
    mc_norm = np.zeros(np.max(cluster_idxs) + 1) # Z in the paper

    for element in elements:
        current_n_clusters = 0
        for cluster in range(cluster_models[element].n_clusters):
            current_n_clusters += 1
            cluster_with_offset = cluster + total_n_clusters
            g_filtered = g_ref[cluster_idxs==cluster_with_offset]
            height_filtered = height[cluster_idxs==cluster_with_offset]
            var_filtered = var[cluster_idxs==cluster_with_offset]

            G_skk = chunked_sum_of_kernels(g_filtered, height_filtered, var_filtered)

            mc_norm[cluster_with_offset] = G_skk / g_filtered.shape[0]

        total_n_clusters += current_n_clusters
    return mc_norm



def distances(P1, p2):
    dv = P1 - p2[None, :]
    d = np.linalg.norm(dv, axis=1)
    return d

def mahalanobis(P1, Var1, p2):
    arg = (P1 - p2[None, :])**2 / Var1
    d = np.sqrt(np.sum(arg, axis=1))
    return d

def combine_kernels(h1, h2, p1, p2, var1, var2):
    h = h1 + h2
    p = (h1 * p1 + h2 * p2) / h
    var = (h1 * (var1 + p1**2) + h2 * (var2 + p2**2))/h - p**2

    return p, h, var


def compress(g, cov, h, thresh=0.8):

    gc = np.full(g.shape, 1000.0)
    covc = np.full(g.shape, 0.0)
    hc = np.full((g.shape[0], 1), 0.0)

    gc[0] = g[0]
    covc[0] = cov[0]
    hc[0] = h[0]

    n_compressed = 1

    for ii in range(1, g.shape[0]):
        P1 = gc[:n_compressed]
        Cov1 = covc[:n_compressed]
        p2 = g[ii]
        h2 = h[ii]
        cov2 = cov[ii]

        dists = mahalanobis(P1,Cov1, p2)
        dmin = np.min(dists)
        idx = np.argmin(dists)

        p1 = gc[idx]
        h1 = hc[idx]
        cov1 = covc[idx]

        if dmin >= thresh:
            gc[n_compressed] = p2
            hc[n_compressed] = h2
            covc[n_compressed] = cov2
            n_compressed += 1
        else:
            pnew, hnew, covnew = combine_kernels(h1, h2, p1, p2, cov1, cov2)

            gc[idx] = pnew
            hc[idx] = hnew
            covc[idx] = covnew
        
    gc = gc[:n_compressed]
    covc = covc[:n_compressed]
    hc = hc[:n_compressed]
    return gc, covc, hc


def incremental_compress(gc, covc, hc, gnew, covnew, hnew, thresh=0.8):
    dists = mahalanobis(gc,covc, gnew)

    dmin = np.min(dists)
    idx = np.argmin(dists)

    p1 = gc[idx]
    h1 = hc[idx]
    cov1 = covc[idx]

    if dmin < thresh:
        pnew, hnew, covnew = combine_kernels(h1, hnew, p1, gnew, cov1, covnew)

        gc[idx] = pnew
        hc[idx] = hnew
        covc[idx] = covnew
    
    else:
        gc = np.append(gc, gnew[None,:], axis=0)
        hc = np.append(hc, hnew[None,:], axis=0)
        covc = np.append(covc, covnew[None,:], axis=0)
    
    return gc, covc, hc