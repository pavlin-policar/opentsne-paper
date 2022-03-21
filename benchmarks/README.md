# Benchmarks

We benchmark *openTSNE* against popular, open-source t-SNE libraries, accross three programming languages, including Python, R, and Julia.

## Running benchmarks

The prerequisite for running the benchmarks is `conda`, which enables us to create reproducable, isolated environments. Run the benchmark scripts (in this same order)

```bash
bash run_python.sh
bash run_r.sh
bash run_julia.sh
```

The benchmark output will be saved to the `logs/` directory.

## Generating figures

Because running the benchmarks can take a long time, we provide the output of our own benchmarks in the `logs/` directory. These benchmarks were run on an Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz processor with 128GB of memory. We also include exact conda environments and installed package versions.

To generate the benchmark figure, run

```bash
python generate_figures.py
```