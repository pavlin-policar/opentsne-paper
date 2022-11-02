#!/bin/bash

set -e

CONDA_ENVIRONMENT_NAME="tsne_benchmarks"

is_installed() {
  [ -x "$(command -v "$1")" ]
}

activate_environment() {
  eval "$(conda shell.bash hook)"
  conda activate "$1"
}

install_fitsne() {
  # Setup FIt-SNE
  git clone https://github.com/KlugerLab/FIt-SNE.git || :
  conda install -c conda-forge fftw -y
  cd FIt-SNE
  CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include/" LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
      g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  \
      -o bin/fast_tsne -pthread -lfftw3 -lm -Wno-address-of-packed-member
  cd ..
}

install_python_dependencies() {
  conda install python numpy scikit-learn -y

  pip install opentsne --no-binary opentsne
  pip install -r requirements-benchmarks.txt

  install_fitsne
}

install_r_dependencies() {
  conda install -c r -c conda-forge r -y
  Rscript -e 'install.packages("Rtsne", repos="https://cloud.r-project.org")'
  Rscript -e 'install.packages("optparse", repos="https://cloud.r-project.org")'
}

install_julia_dependencies() {
  conda install -c conda-forge julia -y
  julia -e 'using Pkg; Pkg.add("ArgParse")'
  julia -e 'using Pkg; Pkg.add("CSV")'
  julia -e 'using Pkg; Pkg.add("DataFrames")'
  julia -e 'using Pkg; Pkg.add("StatsBase")'
  julia -e 'using Pkg; Pkg.add("TSne")'
}

# Download benchmark datasets from remote
# $1: dataset_name -- e.g. macosko_2015, which matches remote file name
download_datasets() {
  mkdir -p data

  wget -nc -P data "http://file.biolab.si/opentsne/benchmark/$1.pkl.gz"
  is_installed python && python convert_pickle_to_csv.py -i "data/$1.pkl.gz" -o "data/$1.csv"
  is_installed Rscript && Rscript convert_csv_to_rds.r -i "data/$1.csv" -o "data/$1.rds"
}

print_usage() {
  cat <<EOF
Run the benchmark suite.
Usage: run.sh [OPTIONS]

Options:
  -s    Run the smaller benchmark suite. The smaller benchmark suite takes
        only a few hours to run, while the full benchmark suite takes days.
  -l    Run the full/large benchmark suite.
  -p    Prepare only. This command will prepare the environment without
        running any of the benchmarks. This needs to be used in conjunction
        to the -s or -f flag, to download the appropriate benchmark
        data sets.
  -h    Print this help message.

EOF
}

# Parse command line options
prepare_only=false
while getopts "hslp" flag; do
  case "$flag" in
    h) print_usage; exit 0;;
    s) small=true;;
    l) small=false;;
    p) prepare_only=true;;
  esac
done

# If neither the full not small benchmarks were specified, exit and print help
if [ -z "$small" ]; then
  print_usage && exit 1
fi

# Configure small/full benchmark suite
if $small; then
  echo "Running small benchmark suite."

  dataset_name="macosko_2015"
  sample_sizes=(1000 2000)
  repetitions=6;

else
  echo "Running full benchmark suite."

  dataset_name="10x_mouse_zheng"
  sample_sizes=(1000 100000 250000 500000 750000 1000000)
  repetitions=6;
fi

# If conda is not installed, there really is nothing to do
if ! is_installed conda; then
  echo -e "The \`conda\` command was not found. Please install \`conda\` and rerun."
  exit 1
fi

# Setup environment if it doesn't already exist
if ! conda env list | grep "\s*$CONDA_ENVIRONMENT_NAME\s*" >/dev/null 2>&1; then
  conda create --name "$CONDA_ENVIRONMENT_NAME" -y
  activate_environment "$CONDA_ENVIRONMENT_NAME"
  install_python_dependencies
  install_r_dependencies
  install_julia_dependencies || true  # allowed to fail on mac
else
  activate_environment "$CONDA_ENVIRONMENT_NAME"
fi

download_datasets $dataset_name

# Prepare logs directory
mkdir -p logs
conda env export | sed '/prefix:/d' > logs/00--conda_env.yml

if $prepare_only; then  # environment configuration complete
  exit 0
fi

# Run Python benchmark suite
# requires environmental variable `cores` to be set
run_python_benchmarks() {
  for method in "${methods[@]}"; do
      for size in "${sample_sizes[@]}"; do
          cmd="OMP_NUM_THREADS=$cores NUMBA_NUM_THREADS=$cores \
              LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
              DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
              python benchmark.py $method run \
              --fname data/$dataset_name.pkl.gz \
              --repetitions $repetitions \
              --n-samples $size \
              --n-jobs $cores 2>&1 \
              | tee -a logs/${dataset_name}--${method}--${cores}_core--${size}_samples.log";
          echo "$cmd" | tr -s " ";
          eval "$cmd";
      done;
  done;
}

# Run R benchmark suite
# requires environmental variable `cores` to be set
run_r_benchmarks() {
  method="Rtsne"
  for size in "${sample_sizes[@]}"; do
    cmd="OMP_NUM_THREADS=$cores \
        LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
        DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib/" \
        Rscript benchmark.r \
        --fname data/$dataset_name.rds \
        --repetitions $repetitions \
        --n-threads $cores \
        --n-samples $size 2>&1 \
        | tee -a logs/${dataset_name}--${method}--${cores}_core--${size}_samples.log";
    echo "$cmd" | tr -s " ";
    eval "$cmd";
  done;
}

# Run Julia benchmark suite
# requires environmental variable `cores` to be set
run_julia_benchmarks() {
  method="TSne-jl"
  for size in "${sample_sizes[@]}"; do
    cmd="OMP_NUM_THREADS=$cores \
        julia benchmark.jl \
        --fname data/$dataset_name.csv \
        --repetitions $repetitions \
        --n-samples $size 2>&1 \
        | tee -a logs/${dataset_name}--${method}--${cores}_core--${size}_samples.log";
    echo "$cmd" | tr -s " ";
    eval "$cmd";
  done;
}


if is_installed python; then
  methods=(openTSNEBH openTSNEFFT MulticoreTSNE FItSNE sklearn UMAP);
  cores=1 run_python_benchmarks;
  cores=8 run_python_benchmarks;
else
  echo -e "The \`python\` command was not found. Skipping."
fi

if is_installed Rscript; then
  cores=1 run_r_benchmarks;
  cores=8 run_r_benchmarks;
else
  echo -e "The \`Rscript\` command was not found. Skipping."
fi

if is_installed julia; then
  sample_sizes=(1000 5000 10000 20000 50000)
  cores=1 run_julia_benchmarks;
else
  echo -e "The \`julia\` command was not found. Skipping."
fi
