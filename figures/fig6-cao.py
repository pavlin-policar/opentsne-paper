import argparse
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
import scanpy as sc

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
adata = sc.read_h5ad(os.path.join(args.in_dir, "cao_2019.h5ad"))

if (
    "X_tsne_exag1" not in adata.obsm or
    "X_tsne_exag2" not in adata.obsm or
    "X_tsne_exag4" not in adata.obsm or
    args.force
):
    # Generate indices for the random sample
    np.random.seed(0)

    x, y = adata.obsm["X_pca"], adata.obs["day"]

    indices = np.random.permutation(list(range(x.shape[0])))
    reverse = np.argsort(indices)

    x_sample, x_rest = x[indices[:25000]], x[indices[25000:]]
    y_sample, y_rest = y[indices[:25000]], y[indices[25000:]]

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
    sample_embedding.optimize(n_iter=250, exaggeration=12, inplace=True)
    sample_embedding.optimize(n_iter=500, exaggeration=1, inplace=True)

    # Calculate full affinities
    print("Calculating full affinities...")
    affinities = openTSNE.affinity.PerplexityBasedNN(
        x,
        perplexity=30,
        metric="cosine",
        n_jobs=24,
        random_state=0,
        verbose=True,
    )

    print("Calculating full initialization...")
    rest_init = sample_embedding.prepare_partial(x_rest, k=1, perplexity=1/3)
    init_full = np.vstack((sample_embedding, rest_init))[reverse]
    init_full = init_full / (np.std(init_full[:, 0]) * 10000)

    # Generate t-SNE Embeddings
    print("Generating embeddings...")
    
    embedding = openTSNE.TSNEEmbedding(
        init_full, affinities, n_jobs=8, random_state=42, verbose=True
    )
    
    embedding_ee = embedding.optimize(n_iter=500, exaggeration=12)
    embedding_exag4 = embedding_ee.optimize(n_iter=500, exaggeration=4)
    embedding_exag2 = embedding_exag4.optimize(n_iter=500, exaggeration=2)
    embedding_exag1 = embedding_exag2.optimize(n_iter=500, exaggeration=1)
    
    # Add computed embeddings to h5ad files and save them to disk
    adata.obsm["X_tsne_ee"] = embedding_ee.view(np.ndarray)
    adata.obsm["X_tsne_exag4"] = embedding_exag4.view(np.ndarray)
    adata.obsm["X_tsne_exag2"] = embedding_exag2.view(np.ndarray)
    adata.obsm["X_tsne_exag1"] = embedding_exag1.view(np.ndarray)
    
    adata.write_h5ad(os.path.join(args.in_dir, "cao_2019.h5ad"))


### Make figure
print("Generating figure...")

fig, ax = plt.subplots(ncols=3, figsize=(16, 16 / 3))

for emb, exag, ax_ in zip(
    [adata.obsm["X_tsne_exag1"], adata.obsm["X_tsne_exag2"], adata.obsm["X_tsne_exag4"]],
    [1, 2, 4],
    ax.ravel()
):
    sc = ax_.scatter(
        emb[:, 0],
        emb[:, 1],
        c=adata.obs["day"],
        cmap="RdYlBu",
        alpha=0.04,
        s=1,
        rasterized=True,
    )
    ax_.set_xticks([]), ax_.set_yticks([])
    ax_.axis("off")
    ax_.axis("equal")
    ax_.set_title(f"exaggeration={exag}", va="baseline")

plt.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)

for ax_, letter in zip(ax.ravel(), string.ascii_lowercase):
    plt.text(
        0, 1.02, letter, transform=ax_.transAxes, fontsize=16, va="baseline", fontweight="bold"
    )

color_bar = fig.colorbar(
    sc,
    ax=ax.ravel().tolist(),
    ticks=adata.obs["day"].unique(),
    label="Day",
    orientation="horizontal",
    anchor=(0.5, 0),
    fraction=0.015,
    aspect=30,
    pad=0.05,
)
color_bar.set_alpha(1)
color_bar.draw_all()

plt.savefig(
    os.path.join(args.out_dir, "cao2019.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "cao2019.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
