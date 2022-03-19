#!/bin/bash


set -e

cat << EOF
================================================================================
Downloading datasets from file.biolab.si...
================================================================================
EOF

# Download data, if not exists
H5AD_DIR="./data/h5ad"
mkdir -p "$H5AD_DIR"

wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/cao_2019.h5ad.tar.gz
wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/harris_2018.h5ad.tar.gz
wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/hochgerner_2018.h5ad.tar.gz
wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/macosko_2015.h5ad.tar.gz
wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/shekhar_2016.h5ad.tar.gz
wget -nc -P "$H5AD_DIR" http://file.biolab.si/opentsne/h5ad/tasic_2018.h5ad.tar.gz

cat << EOF
================================================================================
Extracting files...
================================================================================
EOF

for f in "$H5AD_DIR/*.h5ad.tar.gz"; do
    tar xvfa "$f" --remove-files
done
