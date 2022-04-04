import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn import datasets

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", required=True)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print("Reading data...")
iris = datasets.load_iris()

print("Creating embedding...")
embedding = openTSNE.TSNE().fit(iris.data)

print("Generating figure...")
target = []
for t in iris.target_names[iris.target]:
    target.append("Iris " + t.capitalize())

target = np.array(target)

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
