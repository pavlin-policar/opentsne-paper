import argparse
import sys
from os import path, listdir

import pandas as pd
import scipy.sparse as sp
import anndata
import scanpy as sc
import utils


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

fname = path.join(args.out_dir, "tasic_2018.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

data = pd.read_csv(path.join(args.in_dir, "GSE115746_cells_exon_counts.csv.gz"), index_col=0)

metadata = pd.read_csv(path.join(args.in_dir, "GSE115746_complete_metadata_28706-cells.csv.gz"), index_col=0)

x = sp.csr_matrix(data.values)

adata = anndata.AnnData(
    x.T,
    var={"var_names": data.index.values},
    obs={
        "obs_names": data.columns.values,
        "labels": metadata["cell_class"].astype(str),
        "batch": metadata["source_name"].astype(str),
    },
)

# Load colors
color_metadata = pd.read_csv(path.join(args.in_dir, "tasic-sample_heatmap_plot_data.csv"), index_col=0)

adata_mask = adata.obs_names.isin(color_metadata.index)
adata = adata[adata_mask]

adata.obs["cluster_color"] = color_metadata.loc[adata.obs_names.values, "cluster_color"]

adata.write_h5ad(fname)

print("Running single-cell pipeline...")
sc.pp.filter_genes(adata, min_counts=10)
gene_mask = utils.select_genes(adata.X, n=3000, threshold=0)

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False)

X_pca = utils.pca(adata.X[:, gene_mask])

# Reload the original, unprocessesed data set, and add the PCA coordinates
adata = anndata.read_h5ad(fname)
adata.obsm["X_pca"] = X_pca

adata.write_h5ad(fname)
