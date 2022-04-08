import argparse
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
adata = sc.read_h5ad(os.path.join(args.in_dir, "tasic_2018.h5ad"))

### t-SNE Embeddings
if (
    "X_tsne_perp30" not in adata.obsm or
    "X_tsne_perp500" not in adata.obsm or
    "X_tsne_multiscale" not in adata.obsm or
    "X_tsne_dof06" not in adata.obsm or
    args.force
):
    init = openTSNE.initialization.rescale(adata.obsm["X_pca"][:, :2])
    
    print("Creating embedding perplexity=30...")
    embedding_perp30 = openTSNE.TSNE(
        perplexity=30,
        initialization=init,
        metric="cosine",
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    print("Creating embedding perplexity=30...")
    embedding_perp500 = openTSNE.TSNE(
        perplexity=500,
        initialization=init,
        metric="cosine",
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    print("Creating embedding perplexity=[30, 500]...")
    embedding_multiscale = openTSNE.TSNE(
        perplexity=[30, 500],
        initialization=init,
        metric="cosine",
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    print("Creating embedding dof=0.6...")
    embedding_dof06 = openTSNE.TSNE(
        perplexity=30,
        initialization=init,
        metric="cosine",
        dof=0.6,
        n_jobs=8,
        random_state=0,
        verbose=True,
    ).fit(adata.obsm["X_pca"])
    
    # Add computed embeddings to h5ad files and save them to disk
    adata.obsm["X_tsne_perp30"] = embedding_perp30.view(np.ndarray)
    adata.obsm["X_tsne_perp500"] = embedding_perp500.view(np.ndarray)
    adata.obsm["X_tsne_multiscale"] = embedding_multiscale.view(np.ndarray)
    adata.obsm["X_tsne_dof06"] = embedding_dof06.view(np.ndarray)
    
    adata.write_h5ad(os.path.join(args.in_dir, "tasic_2018.h5ad"))

### Make figure
print("Generating figure...")
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

for ax_, (title, emb) in zip(ax.ravel(), [
    ["perplexity=30", adata.obsm["X_tsne_perp30"]],
    ["perplexity=500", adata.obsm["X_tsne_perp500"]],
    ["perplexity=30,500", adata.obsm["X_tsne_multiscale"]],
    ["dof=0.6", adata.obsm["X_tsne_dof06"]],
]):
    ax_.scatter(emb[:, 0], emb[:, 1], s=1, alpha=1, c=adata.obs["cluster_color"], rasterized=True)
    ax_.set_title(title)

for ax_ in ax.ravel():
    ax_.axis("off")
    ax_.axis("equal")

for ax_, letter in zip(ax.ravel(), string.ascii_lowercase): 
    plt.text(0, 1.02, letter, transform=ax_.transAxes, fontsize=16, va="baseline", fontweight="bold")
    
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=1, bottom=0, left=0, right=1)

plt.savefig(
    os.path.join(args.out_dir, "tasic2018.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "tasic2018.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
