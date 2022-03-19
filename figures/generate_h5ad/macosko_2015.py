import anndata
import argparse
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sys
import utils
from os import path, listdir

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

fname = path.join(args.out_dir, "macosko_2015.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

data = pd.read_table(path.join(args.in_dir, "GSE63472_P14Retina_merged_digital_expression.txt.gz"), index_col=0)

cluster_ids = pd.read_table(path.join(args.in_dir, "retina_clusteridentities.txt"), header=None, index_col=0, squeeze=True)

# Drop unlabelled cells
data = data.loc[:, cluster_ids.index]

# Convert cluster ids to cell type names
cell_types = cluster_ids.astype(object)

cell_types.loc[cell_types == 1] = "Horizontal cells"
cell_types.loc[cell_types == 2] = "Retinal ganglion cells"
cell_types.loc[cell_types.isin(range(3, 24))] = "Amacrine cells"
cell_types.loc[cell_types == 24] = "Rods"
cell_types.loc[cell_types == 25] = "Cones"
cell_types.loc[cell_types.isin(range(26, 34))] = "Bipolar cells"
cell_types.loc[cell_types == 34] = "Muller glia"
cell_types.loc[cell_types == 35] = "Astrocytes"
cell_types.loc[cell_types == 36] = "Fibroblasts"
cell_types.loc[cell_types == 37] = "Vascular endothelium"
cell_types.loc[cell_types == 38] = "Pericytes"
cell_types.loc[cell_types == 39] = "Microglia"

# Create anndata object
x = sp.csr_matrix(data.values)

adata = anndata.AnnData(
    x.T,
    var={"var_names": data.index.values},
    obs={
        "obs_names": data.columns.values,
        "cluster_ids": cluster_ids.values,
        "labels": cell_types.values,
    },
)

adata.write_h5ad(fname)

print("Running single-cell pipeline...")
sc.pp.filter_genes(adata, min_counts=10)
gene_mask = utils.select_genes(adata.X, n=3000, threshold=0)

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
sc.pp.log1p(adata)
sc.pp.scale(adata)

X_pca = utils.pca(adata.X[:, gene_mask])

# Reload the original, unprocessesed data set, and add the PCA coordinates
adata = anndata.read_h5ad(fname)
adata.obsm["X_pca"] = X_pca

adata.write_h5ad(fname)
