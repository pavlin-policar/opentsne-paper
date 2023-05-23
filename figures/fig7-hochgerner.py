import argparse
import os

import numpy as np
import scanpy as sc
import utils
import openTSNE
import matplotlib.pyplot as plt

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
adata = sc.read_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))
new = sc.read_h5ad(os.path.join(args.in_dir, "harris_2018.h5ad"))

### t-SNE Embeddings
if (
    "X_tsne" not in adata.obsm or
    "X_tsne" not in new.obsm or
    args.force
):
    print("Creating reference embedding...")
    init = openTSNE.initialization.rescale(adata.obsm["X_pca"][:, :2])
    
    embedding = openTSNE.TSNE(
        perplexity=[50, 500],
        initialization=init,
        metric="cosine",
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    adata.obsm["X_tsne"] = embedding.view(np.ndarray)

    adata.write_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))

    ### Prepare new data to transform
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.filter_genes(new, min_counts=10)

    ### Align data sets
    print("Aligning data sets...")
    adata.var_names = adata.var_names.str.capitalize()
    new.var_names = new.var_names.str.capitalize()

    # Remove duplicate genes
    adata = adata[:, ~adata.var_names.duplicated()]
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
        embedding.view(np.ndarray),
        affinities,
        n_jobs=8,
        verbose=True,
    )
    new_embedding0 = embedding.prepare_partial(new_1000.X.toarray(), perplexity=30)
    new_embedding = new_embedding0.optimize(250, learning_rate=0.1, exaggeration=10)
    new_embedding = new_embedding.optimize(500, learning_rate=0.1, exaggeration=1.5)

    # Add computed embeddings to h5ad files and save them to disk
    adata.obsm["X_tsne"] = embedding.view(np.ndarray)
    new.obsm["X_tsne"] = new_embedding.view(np.ndarray)

    adata.write_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))
    new.write_h5ad(os.path.join(args.in_dir, "harris_2018.h5ad"))

print("Reloading data...")
adata = sc.read_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))
new = sc.read_h5ad(os.path.join(args.in_dir, "harris_2018.h5ad"))

### Make figure
print("Generating figure...")

colors = {
    # Excitatory neurons
    "immature pyramidal neuron": "#DE6093",
    "pyramidal neuron": "#D5221C",
    "immature granule cell": "#FEE00D",
    "juevnile granule cell": "#FAB508",
    "granule cell": "#FFA02C",
    # Inhibitory neurons
    "immature GABAergic neuron": "#74C9FF",
    "GABAergic neuron": "#68A8E6",
    "Cajal-Retzius cell": "#1C6865",
    # Neural progenitors
    "neuroblast": "#FEA388",
    "neuronal intermediate progenitor cells": "#FF7390",
    # Glia
    "ependymal cell": "#967D78",
    "oligodendrocyte precursor cell": "#9F9F9F",
    "newly formed oligodendrocyte": "#787878",
    "oligodendrocyte": "#505050",
    "immature astrocyte": "#9B7E7B",
    "juvenile astrocyte": "#846F57",
    "astrocyte": "#655C4A",
    "glial cell": "#7C9672",
    "radial glial cell": "#684644",
    
    "endothelial cell": "#5300FF",
    "vascular and leptomeningeal cell": "#8E97E4",
    "perivascular macrophage": "#8655F1",
}

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(16, 8))

ax[0, 0].text(-0, 0.5, "Reference embedding", transform=ax[0, 0].transAxes,
              rotation=90, verticalalignment="center", size="large")
ax[1, 0].text(-0, 0.5, "Embedded samples", transform=ax[1, 0].transAxes,
              rotation=90, verticalalignment="center", size="large")

utils.plot(
    adata.obsm["X_tsne"],
    adata.obs["labels"],
    ax=ax[0, 0],
    s=2,
    colors=colors,
    draw_centers=False,
    draw_cluster_labels=False,
    fontsize=11,
    draw_legend=False,
    legend_kwargs=dict(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        bbox_transform=fig.transFigure,
        labelspacing=1,
        ncol=4,
    )
)

colors_bw = {1: "#CCCCCC"}
utils.plot(adata.obsm["X_tsne"], np.ones_like(adata.obs_names), ax=ax[1, 0],
           colors=colors_bw, alpha=1, s=1, draw_legend=False, zorder=1)
utils.plot(new.obsm["X_tsne"], np.ones_like(new.obs.index), ax=ax[1, 0],
           draw_legend=False, s=1, alpha=1, colors={1: "tab:green"}, zorder=2)

# Plot reference silhouette onto bottom row
for ax_ in ax[1, :].ravel():
    utils.plot(adata.obsm["X_tsne"], np.ones_like(adata.obs_names), ax=ax_,
               colors=colors_bw, alpha=1, s=1, draw_legend=False, zorder=1)

marker_params = dict(binary=True, alpha=1, threshold=1)
# Reference embeddings
utils.plot_marker(["Gad1", "Gad2"], adata, adata.obsm["X_tsne"], ax=ax[0, 1], **marker_params)
utils.plot_marker(["Olig1", "Olig2"], adata, adata.obsm["X_tsne"], ax=ax[0, 2], **marker_params)
utils.plot_marker(["Gfap"], adata, adata.obsm["X_tsne"], ax=ax[0, 3], **marker_params)
# Transformed embeddings
utils.plot_marker(["Gad1", "Gad2"], new, new.obsm["X_tsne"], ax=ax[1, 1], **marker_params)
utils.plot_marker(["Olig1", "Olig2"], new, new.obsm["X_tsne"], ax=ax[1, 2], **marker_params)
utils.plot_marker(["Gfap"], new, new.obsm["X_tsne"], ax=ax[1, 3], **marker_params)

ax[0, 1].set_title(f"Inhibitory Neurons\n{ax[0, 1].get_title()}")
ax[0, 2].set_title(f"Oligodendrocytes\n{ax[0, 2].get_title()}")
ax[0, 3].set_title(f"Astrocytes\n{ax[0, 3].get_title()}")

for ax_ in ax.ravel():
    ax_.set_xticks([]), ax_.set_yticks([])
    ax_.set_axis_off()
    ax_.axis("equal")

for ax_ in ax[1].ravel():
    ax_.set_title("")

fig.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)

plt.savefig(
    os.path.join(args.out_dir, "transform_hochgerner.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "transform_hochgerner.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
