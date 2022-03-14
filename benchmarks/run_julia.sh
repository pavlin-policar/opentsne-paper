#!/bin/bash

set -e

conda create --name tsne_benchmarks_julia -y
eval "$(conda shell.bash hook)"  # activate conda env inside script
conda activate tsne_benchmarks_julia

# Setup Julia environment
conda install -c conda-forge julia -y
julia -e 'using Pkg; Pkg.add("ArgParse")'
julia -e 'using Pkg; Pkg.add("CSV")'
julia -e 'using Pkg; Pkg.add("DataFrames")'
julia -e 'using Pkg; Pkg.add("StatsBase")'
julia -e 'using Pkg; Pkg.add("TSne")'

# Download benchmark dataset
mkdir -p data
wget -nc -P data http://file.biolab.si/opentsne/10x_mouse_zheng.pkl.gz

# Prepare logs directory
mkdir -p logs

# RUN BENCHMARKS
SAMPLE_SIZES=(1000 5000 10000 20000 50000);
REPETITIONS=6;

# Single-threaded benchmarks
for size in ${SAMPLE_SIZES[@]}; do
    cmd="OMP_NUM_THREADS=1 \
        julia benchmark.jl \
        --repetitions $REPETITIONS \
        --n-samples $size 2>&1 \
        | tee -a logs/TSne-jl_${size}.log";
    echo "$cmd";
    eval "$cmd";
done;
