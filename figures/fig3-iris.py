import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn import datasets

import utils

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", required=True)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
iris = datasets.load_iris()

print("Creating embedding...")
embedding = openTSNE.TSNE().fit(iris.data)
new_embedding = embedding.transform(iris.data[::3])

print("Generating figure...")
target = []
for t in iris.target_names[iris.target]:
    target.append("Iris " + t.capitalize())

target = np.array(target)

# Original embedding
fig, ax = plt.subplots(figsize=(5, 5))
utils.plot(embedding, target, s=64, alpha=0.75, ax=ax)
ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
ax.axis("on")
plt.savefig(
    os.path.join(args.out_dir, "iris.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "iris.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.close()

# Transform embedding
fig, ax = plt.subplots(figsize=(5, 5))
utils.plot(embedding, target, s=32, alpha=0.25, ax=ax)
utils.plot(new_embedding, target[::3], s=64, alpha=1, ax=ax)
ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
ax.axis("on")
plt.savefig(
    os.path.join(args.out_dir, "iris-transform.pdf"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.savefig(
    os.path.join(args.out_dir, "iris-transform.png"),
    dpi=72,
    bbox_inches="tight",
    transparent=True,
)
plt.close()
