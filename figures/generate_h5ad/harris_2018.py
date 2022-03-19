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

fname = path.join(args.out_dir, "harris_2018.h5ad")

if path.exists(fname) and not args.force:
    print(f"`{fname}` exists. Skipping...")
    sys.exit(0)

data = pd.read_table(path.join(args.in_dir, "GSE99888_gene_expression.tab.gz"), index_col=0)

# Remove duplicate cells
data = data[~data.index.duplicated()]


x = sp.csr_matrix(data.values)

adata = anndata.AnnData(
    x.T,
    var={"var_names": data.index.values},
    obs={"obs_names": data.columns.values},
)

adata.write_h5ad(fname)
