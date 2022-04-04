import argparse
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
import scanpy as sc

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
adata = sc.read_h5ad(os.path.join(args.in_dir, "macosko_2015.h5ad"))
new = sc.read_h5ad(os.path.join(args.in_dir, "shekhar_2016.h5ad"))

### t-SNE Embeddings
if (
    "X_tsne_standard" not in adata.obsm or
    "X_tsne_modern" not in adata.obsm or
    "X_tsne_transform" not in new.obsm or
    args.force
):
    # Standard embedding
    print("Creating standard embedding...")
    embedding_standard = openTSNE.TSNE(
        perplexity=30,
        initialization="random",
        metric="cosine",
        learning_rate=200,
        n_iter=750,
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    # Multiscale embedding
    print("Creating multiscale embedding...")
    embedding_modern = openTSNE.TSNE(
        perplexity=[50, 500],
        initialization="pca",
        metric="cosine",
        learning_rate="auto",
        n_iter=500,
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    ### Prepare new data to transform
    # Synchronize labels
    new.obs["labels"] = new.obs["labels"].astype(str)
    new.obs["labels"][new.obs["labels"].str.contains("bipolar cell")] = "retinal bipolar neuron"
    new.obs["labels"] = new.obs["labels"].replace({
        "retinal rod cell": "Rods",
        "retinal bipolar neuron": "Bipolar cells",
        "amacrine cell": "Amacrine cells",
        "retinal cone cell": "Cones",
        "Mueller cell": "Muller glia",
        "retinal ganglion cell": "Retinal ganglion cells",
        "endothelial cell": "Vascular endothelium",
        "retina horizontal cell": "Horizontal cells",
        "fibroblast": "Fibroblasts",
        "microglial cell": "Microglia",
        "pericyte cell": "Pericytes",
        "astrocyte": "Astrocytes",
    })
    
    ### Align data sets
    print("Aligning data sets...")
    adata.var_names = adata.var_names.str.upper()
    new.var_names = new.var_names.str.upper()
    
    # Remove duplicate genes
    new = new[:, ~new.var_names.duplicated()]
    
    # Find shared genes
    shared_genes = adata.var_names[adata.var_names.isin(new.var_names)]
    
    # Keep only shared genes
    adata = adata[:, adata.var_names.isin(shared_genes)]
    new = new[:, new.var_names.isin(shared_genes)]
    
    # Make sure the gene order is the same in both data sets
    adata = adata[:, adata.var_names.argsort()].copy()
    new = new[:, new.var_names.argsort()].copy()
    assert all(adata.var_names == new.var_names)
    
    # Select informative genes, this time considering only the shared genes
    gene_mask = utils.select_genes(adata.X, n=1000, threshold=0)
    
    # Select subsets
    adata_1000 = adata[:, gene_mask]
    new_1000 = new[:, gene_mask]

    ### Embed new data into existing embedding
    print("Transforming samples into embedding...")
    affinities = openTSNE.affinity.PerplexityBasedNN(
        adata_1000.X.toarray(),
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=0,
        verbose=True,
    )
    embedding = openTSNE.TSNEEmbedding(
        embedding_modern.view(np.ndarray),
        affinities,
        n_jobs=8,
        verbose=True,
    )
    new_embedding = embedding.transform(
        new_1000.X.toarray(),
        learning_rate=0.1,
        n_iter=500,
        exaggeration=100,
    )
    new_embedding.optimize(500, exaggeration=1, learning_rate=0.1, inplace=True)
    
    # Add computed embeddings to h5ad files and save them to disk
    adata.obsm["X_tsne_standard"] = embedding_standard.view(np.ndarray)
    adata.obsm["X_tsne_modern"] = embedding_modern.view(np.ndarray)
    new.obsm["X_tsne_transform"] = new_embedding.view(np.ndarray)
    
    adata.write_h5ad(os.path.join(args.in_dir, "macosko_2015.h5ad"))
    new.write_h5ad(os.path.join(args.in_dir, "shekhar_2016.h5ad"))

### Make figure
print("Generating figure...")
    
colors = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}

cluster_ids = np.array(adata.obs["cluster_ids"], dtype=float).astype(int)

cluster_cell_mapping = {
    1: "Horizontal cells",
    2: "Retinal ganglion cells",
    24: "Rods",
    25: "Cones",
    34: "Muller glia",
    35: "Astrocytes",
    36: "Fibroblasts",
    37: "Vascular endothelium",
    38: "Pericytes",
    39: "Microglia",
}
for i in range(3, 24):
    cluster_cell_mapping[i] = "Amacrine cells"
for i in range(26, 34):
    cluster_cell_mapping[i] = "Bipolar cells"
    
# Rename clusters
cluster_ids_ = [cluster_cell_mapping[yi] if yi in (1, 2, 24, 25, 34, 35, 36, 37, 38, 39) else yi 
                for yi in cluster_ids]
cluster_ids_ = np.array(cluster_ids_)

# Create color mapping for the renamed clusters
colors_ = {}
for label in np.unique(cluster_ids_):
    try:
        assert int(label) in cluster_cell_mapping
        colors_[label] = colors[cluster_cell_mapping[int(label)]]
    except ValueError:
        colors_[label] = colors[label]

# Final figure
fig, ax = plt.subplots(ncols=3, figsize=(16, 16 / 3))

utils.plot(
    adata.obsm["X_tsne_standard"] @ utils.rotate(-120) * [-1, 1],
    cluster_ids_,
    colors=colors_,
    ax=ax[0],
    fontsize=10,
    draw_centers=True,
    draw_cluster_labels=True,
    draw_legend=False,
    rasterized=True,
)
ax[0].set_title("Original t-SNE")

utils.plot(
    adata.obsm["X_tsne_modern"],
    cluster_ids_,
    colors=colors_,
    ax=ax[1],
    fontsize=10,
    draw_centers=True,
    draw_cluster_labels=True,
    draw_legend=False,
    rasterized=True,
)
ax[1].set_title("Modern t-SNE")

colors_bw = {1: "#666666"}
utils.plot(
    adata.obsm["X_tsne_modern"],
    np.ones_like(adata.obs["labels"]),
    colors=colors_bw,
    ax=ax[2],
    draw_centers=False,
    draw_legend=False,
    alpha=0.05,
    rasterized=True,
)
utils.plot(
    new.obsm["X_tsne_transform"],
    new.obs["labels"],
    colors=colors,
    ax=ax[2],
    rasterized=True,
    draw_centers=False,
    draw_legend=False,
    alpha=0.1,
    s=6,
    lw=0,
)
ax[2].set_title("Embedding New Samples")

for ax_ in ax.ravel():
    ax_.axis("equal")
    
ax[2].set_xlim(ax[1].get_xlim()), ax[2].set_ylim(ax[1].get_ylim())

plt.text(38, -12.5, "Amacrine\ncells", color="k", fontsize=10, horizontalalignment="center", transform=ax[0].transData)
plt.text(27, 30, "Bipolar\ncells", color="k", fontsize=10, horizontalalignment="center", transform=ax[0].transData)

plt.text(32, 24, "Amacrine\ncells", color="k", fontsize=10, horizontalalignment="center", transform=ax[1].transData)
plt.text(30, -21.5, "Bipolar\ncells", color="k", fontsize=10, horizontalalignment="center", transform=ax[1].transData)

for ax_, letter in zip(ax.ravel(), string.ascii_lowercase):
    plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=16, va="baseline", fontweight="bold")
    
fig.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)

plt.savefig(
    os.path.join(args.out_dir, "macosko2015.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "macosko2015.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
