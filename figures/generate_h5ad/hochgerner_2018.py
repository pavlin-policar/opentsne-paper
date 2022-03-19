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

fname = path.join(args.out_dir, "hochgerner_2018.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

data = pd.read_table(path.join(args.in_dir, "GSE104323_10X_expression_data_V2.tab.gz"), index_col=0)

metadata = pd.read_table(path.join(args.in_dir, "GSE104323_metadata_barcodes_24185cells.txt.gz"), index_col=0)

# Remove the cells for which we don't have the appropriate metadata
nan_mask = metadata.isna().all(axis=1)
metadata = metadata.loc[~nan_mask]
data = data.loc[:, metadata.index]


cell_types = metadata["characteristics: cell cluster"].copy()

cell_types = cell_types.replace({
    "Immature-Pyr": "immature pyramidal neuron",
    "GC-juv": "juevnile granule cell",
    "GC-adult": "granule cell",
    "Immature-GC": "immature granule cell",
    "Neuroblast": "neuroblast",
    "Astro-adult": "astrocyte",
    "Immature-GABA": "immature GABAergic neuron",
    "Astro-juv": "juvenile astrocyte",
    "OPC": "oligodendrocyte precursor cell",
    "MOL": "oligodendrocyte",
    "RGL_young": "radial glial cell",
    "Immature-Astro": "immature astrocyte",
    "Endothelial": "endothelial cell",
    "Cajal-Retzius": "Cajal-Retzius cell",
    "CA3-Pyr": "pyramidal neuron",
    "nIPC-perin": "neuronal intermediate progenitor cells",
    "nIPC": "neuronal intermediate progenitor cells",
    "MiCajal-Retziusoglia": "glial cell",
    "NFOL": "newly formed oligodendrocyte",
    "GABA": "GABAergic neuron",
    "RGL": "radial glial cell",
    "Ependymal": "ependymal cell",
    "VLMC": "vascular and leptomeningeal cell",
    "PVM": "perivascular macrophage",
})


x = sp.csr_matrix(data.values)

adata = anndata.AnnData(
    x.T,
    var={"var_names": data.index.values},
    obs={
        "obs_names": data.columns.values,
        "labels": cell_types.values.astype(str),
        "age": metadata["characteristics: age"].values,
    },
)

adata.write_h5ad(fname)

print("Running single-cell pipeline...")
sc.pp.filter_genes(adata, min_counts=10)
gene_mask = utils.select_genes(adata.X, n=1000, threshold=0)

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False)

X_pca = utils.pca(adata.X[:, gene_mask])

# Reload the original, unprocessesed data set, and add the PCA coordinates
adata = anndata.read_h5ad(fname)
adata.obsm["X_pca"] = X_pca

adata.write_h5ad(fname)
