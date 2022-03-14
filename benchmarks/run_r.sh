#!/bin/bash

set -e

conda create --name tsne_benchmarks_r -y
eval "$(conda shell.bash hook)"  # activate conda env inside script
conda activate tsne_benchmarks_r

# Setup R environment
mamba install -c r r -y
Rscript -e 'install.packages("Rtsne", repos="https://cloud.r-project.org")'
Rscript -e 'install.packages("optparse", repos="https://cloud.r-project.org")'

# Download benchmark dataset
mkdir -p data
wget -nc -P data http://file.biolab.si/opentsne/10x_mouse_zheng.pkl.gz

mamba install -y python numpy
python convert_pickle_to_csv.py

# Prepare logs directory
mkdir -p logs

conda list > 00--run_r--conda_list.txt

# RUN BENCHMARKS
REPETITIONS=6;

# Single-threaded benchmarks
SAMPLE_SIZES=(1000 100000 250000 500000 750000);
for size in ${SAMPLE_SIZES[@]}; do
    cmd="OMP_NUM_THREADS=1 \
       Rscript benchmark.r \
        --repetitions $REPETITIONS \
        --n-threads 1 \
        --n-samples $size 2>&1 \
        | tee -a logs/Rtsne_${size}.log";
    echo "$cmd";
    eval "$cmd";
done;

# Multi-threaded benchmarks
SAMPLE_SIZES=(1000 100000 250000 500000 750000 1000000);
for size in ${SAMPLE_SIZES[@]}; do
    cmd="OMP_NUM_THREADS=8 \
        Rscript benchmark.r \
        --repetitions $REPETITIONS \
        --n-threads 8 \
        --n-samples $size 2>&1 \
        | tee -a logs/Rtsne_8core_${size}.log";
    echo "$cmd";
    eval "$cmd";
done;
