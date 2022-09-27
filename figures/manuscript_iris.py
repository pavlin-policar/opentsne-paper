"""
This code is used in the manuscript and demonstrates the usage of openTSNE
on a simple example. The recommended way to use this is to run ipython and
paste in the code line-by-line. This way, the different types of objects and
flow will be most apparent.
"""
import openTSNE
openTSNE.__version__

# Simple usage
from sklearn import datasets

iris = datasets.load_iris()
iris.data[:3]

import openTSNE
embedding = openTSNE.TSNE().fit(iris.data)
embedding[:3]

new_data = iris.data[::3]
new_embedding = embedding.transform(new_data)
new_embedding[:3]

# Advanced usage
from sklearn import datasets

iris = datasets.load_iris()
iris.data[:3]

import openTSNE

affinities = openTSNE.affinity.PerplexityBasedNN(
    iris.data, perplexity=30
)

initialization = openTSNE.initialization.pca(iris.data)
initialization[:3]

embedding = openTSNE.TSNEEmbedding(initialization, affinities)
embedding[:3]

embedding.optimize(250, exaggeration=12, momentum=0.5, inplace=True)
embedding.optimize(500, momentum=0.8, inplace=True)
embedding[:3]

import numpy as np
np.linalg.norm(openTSNE.TSNE().fit(iris.data) - embedding)

new_embedding = embedding.prepare_partial(iris.data[::3], perplexity=5)
new_embedding.optimize(
    exaggeration=1.5, n_iter=250, learning_rate=0.1, momentum=0.8,
    max_grad_norm=0.25, inplace=True
)
new_embedding[::3]

np.linalg.norm(new_embedding - embedding.transform(iris.data[::3]))
