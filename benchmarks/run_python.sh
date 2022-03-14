#!/bin/bash

set -e

conda create --name tsne_benchmarks -y
eval "$(conda shell.bash hook)"  # activate conda env inside script
conda activate tsne_benchmarks
conda install -y python numpy scikit-learn

pip install opentsne
pip install -r requirements-benchmarks.txt

# MulticoreTSNE forces lower version of numpy
pip install --upgrade numpy

# Setup FIt-SNE
git clone git@github.com:KlugerLab/FIt-SNE.git || :
conda install -c conda-forge fftw -y
cd FIt-SNE
CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include/" LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
    g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  \
    -o bin/fast_tsne -pthread -lfftw3 -lm -Wno-address-of-packed-member
cd ..

# Fetch benchmarking module from openTSNE
wget -nc https://raw.githubusercontent.com/pavlin-policar/openTSNE/master/benchmarks/benchmark.py
# Download benchmark dataset
mkdir -p data
wget -nc -P data http://file.biolab.si/opentsne/10x_mouse_zheng.pkl.gz

# Prepare logs directory
mkdir -p logs

# RUN BENCHMARKS
SAMPLE_SIZES=(1000 100000 250000 500000 750000 1000000);
REPETITIONS=6;

# Single-threaded benchmarks
METHODS=(openTSNEBH openTSNEFFT MulticoreTSNE FItSNE sklearn UMAP);

for method in ${METHODS[@]}; do
    for size in ${SAMPLE_SIZES[@]}; do
        cmd="OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
            python benchmark.py $method run_multiple \
            --n $REPETITIONS \
            --n-samples $size 2>&1 \
            | tee -a logs/${method}_${size}.log";
        echo "$cmd";
        eval "$cmd";
    done;
done;

# Multi-threaded benchmarks
METHODS=(openTSNEBH_8core openTSNEFFT_8core MulticoreTSNE_8core FItSNE_8core sklearn_8core UMAP_8core);
for method in ${METHODS[@]}; do
    for size in ${SAMPLE_SIZES[@]}; do
        cmd="OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
            python benchmark.py $method run_multiple \
            --n $REPETITIONS \
            --n-samples $size 2>&1 \
            | tee -a logs/${method}_${size}.log";
        echo "$cmd";
        eval "$cmd";
    done;
done;
