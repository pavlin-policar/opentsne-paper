import argparse
import os

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
adata = sc.read_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))

### t-SNE Embeddings
if (
    "X_tsne" not in adata.obsm or
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
    ).fit(adata.obsm["X_pca"], initialization=init)
    
    adata.obsm["X_tsne"] = embedding.view(np.ndarray)

    adata.write_h5ad(os.path.join(args.in_dir, "hochgerner_2018.h5ad"))

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
