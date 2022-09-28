import gzip
import pickle
import signal
import time
from contextlib import contextmanager

import fire
import numpy as np
from sklearn.manifold import TSNE as SKLTSNE
from sklearn.utils import check_random_state

import openTSNE
import openTSNE.callbacks


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def _timeout(time):
    """Limit runtime of a particular chunk of code.

    Parameters
    ----------
    time: int
        Time in minutes

    """
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(int(time * 60))

    try:
        yield
    except TimeoutError:
        raise TimeoutError(f"Function call timed out after {time} minutes!")
    finally:
        # Unregister the signal, so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


class TSNEBenchmark:
    perplexity = 30
    learning_rate = 200

    def run(self, fname: str, n_samples=1000, repetitions=1, n_jobs=1):
        x, y = self.load_data(fname=fname, n_samples=n_samples)

        for idx in range(repetitions):
            self._run(x, random_state=idx, n_jobs=n_jobs)

    def _run(self, data, random_state=None, n_jobs=1):
        raise NotImplementedError()

    def load_data(self, fname: str, n_samples: int = None):
        with gzip.open(fname, "rb") as f:
            data = pickle.load(f)

        x, y = data["pca_50"], data["CellType1"]
        print(f"Full data set dimensions: {x.shape}")

        if n_samples is not None:
            indices = np.random.choice(
                list(range(x.shape[0])), n_samples, replace=False
            )
            x, y = x[indices], y[indices]

        print(f"Benchmark data set dimensions: {x.shape}")
        return x, y


class openTSNENNDescent(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        print("-" * 80)
        print(f"openTSNE v{openTSNE.__version__}")
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="pynndescent",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=True,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state, verbose=True
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=n_jobs,
            negative_gradient_method="fft",
            random_state=random_state,
            verbose=True,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start, flush=True)


class openTSNEBH(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        print("-" * 80)
        print(f"openTSNE v{openTSNE.__version__}")
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="annoy",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=True,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state, verbose=True
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=n_jobs,
            negative_gradient_method="bh",
            random_state=random_state,
            verbose=True,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start, flush=True)


class openTSNEFFT(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        print("-" * 80)
        print(f"openTSNE v{openTSNE.__version__}")
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="annoy",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=True,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state, verbose=True,
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=n_jobs,
            negative_gradient_method="fft",
            random_state=random_state,
            verbose=True,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start, flush=True)


class MulticoreTSNE(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        from MulticoreTSNE import MulticoreTSNE as MulticoreTSNE_

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        start = time.time()
        tsne = MulticoreTSNE_(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            perplexity=self.perplexity,
            n_jobs=n_jobs,
            angle=0.5,
            verbose=True,
            random_state=random_state,
        )
        tsne.fit_transform(x)
        print("MulticoreTSNE:", time.time() - start, flush=True)


class FItSNE(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        import sys;
        sys.path.append("FIt-SNE")
        from fast_tsne import fast_tsne

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        if random_state == -1:
            init = openTSNE.initialization.random(x, n_components=2)
        else:
            init = openTSNE.initialization.random(
                x, n_components=2, random_state=random_state
            )

        start = time.time()
        fast_tsne(
            x,
            map_dims=2,
            initialization=init,
            perplexity=self.perplexity,
            stop_early_exag_iter=250,
            max_iter=1000,
            early_exag_coeff=12,
            nthreads=n_jobs,
            seed=random_state,
        )
        print("FItSNE:", time.time() - start, flush=True)


class sklearn(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start = time.time()
        SKLTSNE(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            angle=0.5,
            perplexity=self.perplexity,
            init=init,
            verbose=True,
            random_state=random_state,
            n_jobs=n_jobs,
        ).fit_transform(x)
        print("sklearn:", time.time() - start, flush=True)


class UMAP(TSNEBenchmark):
    def _run(self, x, random_state=None, n_jobs=1):
        import umap

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        start = time.time()
        umap.UMAP(random_state=random_state).fit_transform(x)
        print("UMAP:", time.time() - start, flush=True)


if __name__ == "__main__":
    fire.Fire()
