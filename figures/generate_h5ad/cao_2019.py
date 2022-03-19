import anndata
import argparse
import numpy as np
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

fname = path.join(args.out_dir, "cao_2019.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

adata = anndata.read_mtx(path.join(args.in_dir, "GSE119945_gene_count.txt.gz"))
adata = adata.T

cell_annotations = pd.read_csv(path.join(args.in_dir, "GSE119945_cell_annotate.csv.gz"))
adata.obs = cell_annotations.set_index("sample")

gene_annotations = pd.read_csv(path.join(args.in_dir, "GSE119945_gene_annotate.csv.gz"))
adata.var = gene_annotations.set_index("gene_short_name")

# Select main cluster
adata.obs = adata.obs.rename(columns={"Main_Cluster": "cluster_id"})

adata.obs["potential_doublet_cluster"] = adata.obs["potential_doublet_cluster"].astype(str)
adata.obs["detected_doublet"] = adata.obs["detected_doublet"].astype(str)

# Removing putative doublets
print("Removing %d putative doublets" % np.sum(adata.obs["detected_doublet"] == "nan"))
adata = adata[adata.obs["detected_doublet"] != "nan"]

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
