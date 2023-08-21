# Benchmarks

We benchmark *openTSNE* against popular, open-source t-SNE libraries, accross three programming languages, including Python, R, and Julia.

## Running benchmarks

The prerequisite for running the benchmarks is `conda`, which must be previously installed. `conda` enables us to create reproducable, isolated environments. You can run the full benchmark suite using

```bash
bash run.sh -l
```

Alternatively, since the full benchmark suite can take days or even weeks to complete, you may instead wish to run the smaller benchmark suite using

```bash
bash run.sh -s
```

Note, however, that the strength of `openTSNE` over other implementations is its ability to quickly create embeddings of massive data sets. As such, the smaller benchmark suite will fail to highlight the scale of the advantage of `openTSNE` to other implementations.

The benchmark output will be saved to the `logs/` directory. We also include exact [conda](https://docs.conda.io/en/latest/miniconda.html) environment used to produce the benchmarks in the manuscript. This can also be found in the `logs/` directory, and can be reproduced exactly using

```bash
conda env create -f logs/00--conda_env.yml
```

**WARNING**: Please note that Julia is not available via conda on OSX systems. To replicate the environment on an OSX system, please modify `logs/00--conda_env.yml` and remove the `julia` package from the `environment` section. The Julia benchmarks will not be run, but the Python and R benchmarks will be unaffected.

## Generating figures

Because running the benchmarks can take a long time, we provide the output of our own benchmarks in the `logs/` directory. These benchmarks were run on an Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz processor with 128GB of memory, and we include the output of these benchmarks in the `intel_xeon_e5_1650` folder. We also include the exact `conda` environment and installed package versions.

To generate the benchmark figure, install the requirements listed in `requirements-figures.txt`

```bash
pip install -r requirements-figures.txt
```

and run

```bash
python generate_figures.py -i logs/
```