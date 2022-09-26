"""
To run benchmarks in non-Python programming languages, we need to convert the
pickle file to something these languages can read. We choose CSV.

The pickle file must adhere to the format:
{
    "pca_50": np.ndarray  # the data matrix,
    ...
}

"""
import argparse
import gzip
import numpy as np
import os
import pickle
import sys

parser = argparse.ArgumentParser(description="Convert a .pkl.gz file to csv")
parser.add_argument("-i", "--input", help="Input file", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("-f", "--force", help="Force overwrite output file", action="store_true")
args = parser.parse_args()

if os.path.exists(args.output) and not args.force:
    print(f"File `{args.output}` exists. Doing nothing.")
    sys.exit(0)

with gzip.open(args.input, "rb") as f:
    data = pickle.load(f)

np.savetxt(args.output, data["pca_50"])
