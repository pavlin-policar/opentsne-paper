import argparse
import os
import gzip
import pickle

import numpy as np
import scanpy as sc
import utils
import openTSNE
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
adata = sc.read_h5ad(os.path.join(args.in_dir, "cao_2019.h5ad"))

# Generate indices for the random sample
np.random.seed(0)

x, y = adata.obsm["X_pca"], adata.obs["day"]

indices = np.random.permutation(list(range(x.shape[0])))
reverse = np.argsort(indices)
    
x_sample, x_rest = x[indices[:25000]], x[indices[25000:]]
y_sample, y_rest = y[indices[:25000]], y[indices[25000:]]


# Create sample embedding
if not os.path.exists("_cache/cao_sample_embedding.pkl.gz"):
    print("Creating sample embedding...")
    
    sample_affinities = openTSNE.affinity.PerplexityBasedNN(
        x_sample,
        perplexity=500,
        metric="cosine",
        n_jobs=24,
        random_state=0,
        verbose=True,
    )
    sample_init = openTSNE.initialization.spectral(sample_affinities.P)
    
    sample_embedding = openTSNE.TSNEEmbedding(
        sample_init, sample_affinities, n_jobs=24, verbose=True,
    )
    sample_embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    sample_embedding.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)
    
    if not os.path.exists("_cache"):
        os.mkdir("_cache")
        
    with gzip.open(os.path.join("_cache", "cao_sample_embedding.pkl.gz"), "wb") as f:
        pickle.dump({"embedding": sample_embedding, "y": y_sample}, f)

else:
    print("Using precomputed sample embedding...")
    with gzip.open(os.path.join("_cache", "cao_sample_embedding.pkl.gz"), "rb") as f:
        tmp = pickle.load(f)
        sample_embedding, y_sample = tmp["embedding"], tmp["y"]


# Calculate full affinities
#if not os.path.exists("_cache/cao_affinities_full.pkl.gz"):
print("Calculating full affinities...")

affinities = openTSNE.affinity.PerplexityBasedNN(
    x,
    perplexity=30,
    metric="cosine",
    n_jobs=24,
    random_state=0,
    verbose=True,
)
    
#    with gzip.open(os.path.join("_cache", "cao_affinities_full.pkl.gz"), "wb") as f:
#        pickle.dump(affinities, f)

#else:
#    print("Using precomputed full affinities...")
#    with gzip.open(os.path.join("_cache", "cao_affinities_full.pkl.gz"), "rb") as f:
#        affinities = pickle.load(f)


# Calculate full initialization
if not os.path.exists("_cache/cao_init_full.pkl.gz"):
    print("Calculating full initialization...")
    
    rest_init = sample_embedding.prepare_partial(x_rest, k=1, perplexity=1/3)
    init_full = np.vstack((sample_embedding, rest_init))[reverse]
    
    init_full = init_full / (np.std(init_full[:, 0]) * 10000)
        
    with gzip.open(os.path.join("_cache", "cao_init_full.pkl.gz"), "wb") as f:
        pickle.dump(init_full, f)

else:
    print("Using precomputed full initialization...")
    with gzip.open(os.path.join("_cache", "cao_init_full.pkl.gz"), "rb") as f:
        init_full = pickle.load(f)
       
# Generate t-SNE Embeddings 
if (
    "X_tsne_exag1" not in adata.obsm or
    "X_tsne_exag2" not in adata.obsm or
    "X_tsne_exag4" not in adata.obsm or
    args.force
):
    print("Generating embeddings...")
    
    embedding = openTSNE.TSNEEmbedding(
        init_full, affinities, n_jobs=8, random_state=42, verbose=True
    )
    
    embedding_ee = embedding.optimize(n_iter=500, exaggeration=12, momentum=0.5)
    embedding_exag4 = embedding_ee.optimize(n_iter=500, exaggeration=4, momentum=0.8)
    embedding_exag2 = embedding_exag4.optimize(n_iter=500, exaggeration=2, momentum=0.8)
    embedding_exag1 = embedding_exag2.optimize(n_iter=500, exaggeration=1, momentum=0.8)
    
    # Add computed embeddings to h5ad files and save them to disk
    adata.obsm["X_tsne_ee"] = embedding_ee.view(np.ndarray)
    adata.obsm["X_tsne_exag4"] = embedding_exag4.view(np.ndarray)
    adata.obsm["X_tsne_exag2"] = embedding_exag2.view(np.ndarray)
    adata.obsm["X_tsne_exag1"] = embedding_exag1.view(np.ndarray)
    
    adata.write_h5ad(os.path.join(args.in_dir, "cao_2019.h5ad"))
