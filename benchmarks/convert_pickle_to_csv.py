"""
To run benchmarks in non-Python programming languages, we need to convert the
pickle file to something these languages can read. We choose CSV.
"""
import gzip
import pickle
from os import path

import numpy as np

CSV_FNAME = path.join("data", "10x_mouse_zheng.csv")

if not path.exists(CSV_FNAME):
    with gzip.open(path.join("data", "10x_mouse_zheng.pkl.gz"), "rb") as f:
        data = pickle.load(f)

    np.savetxt(CSV_FNAME, data["pca_50"])
