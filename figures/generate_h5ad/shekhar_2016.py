import anndata
import argparse
import pandas as pd
import scipy.sparse as sp
import sys
from os import path, listdir

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in-dir", required=True)
parser.add_argument("-o", "--out-dir", required=True)
parser.add_argument("-f", "--force", action="store_true")
args = parser.parse_args()

fname = path.join(args.out_dir, "shekhar_2016.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

data = pd.read_table(path.join(args.in_dir, "GSE81904_BipolarUMICounts_Cell2016.txt.gz"), index_col=0)

metadata = pd.read_table(path.join(args.in_dir, "clust_retinal_bipolar.txt"), index_col=0)

data = data.loc[:, metadata.index.values]
metadata = metadata.loc[data.columns.values]


cell_types = metadata["SUB-CLUSTER"].astype(object)

cell_types[cell_types.str.contains("BC1")] = "type 1 cone bipolar cell"
cell_types[cell_types.str.contains("BC2")] = "type 2 cone bipolar cell"
cell_types[cell_types.str.contains("BC3")] = "type 3 cone bipolar cell"
cell_types[cell_types.str.contains("BC4")] = "type 4 cone bipolar cell"
cell_types[cell_types.str.contains("BC5")] = "type 5 cone bipolar cell"
cell_types[cell_types.str.contains("BC6")] = "type 6 cone bipolar cell"
cell_types[cell_types.str.contains("BC7")] = "type 7 cone bipolar cell"

# This is a mixture, so go up one level in the ontology
cell_types[cell_types == "BC8/9 (mixture of BC8 and BC9)"] = "type 8/9 cone bipolar cell"

# Non-bipolar cells
cell_types[cell_types == "RBC (Rod Bipolar cell)"] = "rod bipolar cell"
cell_types[cell_types == "MG (Mueller Glia)"] = "Mueller cell"
cell_types[cell_types == "AC (Amacrine cell)"] = "amacrine cell"
cell_types[cell_types == "Rod Photoreceptors"] = "retinal rod cell"
cell_types[cell_types == "Cone Photoreceptors"] = "retinal cone cell"


# Remove all the doublet/contaminants
contaminant_mask = cell_types == "Doublets/Contaminants"
metadata = metadata.iloc[~contaminant_mask.values]
cell_types = cell_types[~contaminant_mask]
data = data.iloc[:, ~contaminant_mask.values]


x = sp.csr_matrix(data.values)

adata = anndata.AnnData(
    x.T,
    var={"var_names": data.index.values},
    obs={
        "obs_names": data.columns.values,
        "labels": cell_types.values,
    },
)

adata.write_h5ad(fname)
