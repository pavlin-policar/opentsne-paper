#!/bin/bash

# This script downloads the raw files from NCBI and converts them to h5ad files.
# This can take a very long time, and we recommend downloading the files from
# file.biolab.si/opentsne/ instead using the `download_h5ad.sh` script.

set -e

RAW_DIR="./data/raw"
H5AD_DIR="./data/h5ad"

cat << EOF
================================================================================
Downloading datasets...
================================================================================
EOF

# Download data, if not exists
mkdir -p "$RAW_DIR"

# Cao 2019
wget -nc -P "$RAW_DIR/cao_2019" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119945/suppl/GSE119945_gene_count.txt.gz
wget -nc -P "$RAW_DIR/cao_2019" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119945/suppl/GSE119945_cell_annotate.csv.gz
wget -nc -P "$RAW_DIR/cao_2019" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119945/suppl/GSE119945_gene_annotate.csv.gz

# Macosko 2015
wget -nc -P "$RAW_DIR/macosko_2015" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63472/suppl/GSE63472%5FP14Retina%5Fmerged%5Fdigital%5Fexpression%2Etxt%2Egz
wget -nc -P "$RAW_DIR/macosko_2015" http://mccarrolllab.org/wp-content/uploads/2015/05/retina_clusteridentities.txt

# Shekhar 2016
wget -nc -P "$RAW_DIR/shekhar_2016" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81904/suppl/GSE81904_BipolarUMICounts_Cell2016.txt.gz
wget -nc -P "$RAW_DIR/shekhar_2016" https://scrnaseq-public-datasets.s3.amazonaws.com/manual-data/shekhar/clust_retinal_bipolar.txt

# Harris 2018
wget -nc -P "$RAW_DIR/harris_2018" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99888/suppl/GSE99888_gene_expression.tab.gz

# Tasic 2018
wget -nc -P "$RAW_DIR/tasic_2018" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcells%5Fexon%5Fcounts%2Ecsv%2Egz
wget -nc -P "$RAW_DIR/tasic_2018" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcomplete%5Fmetadata%5F28706%2Dcells%2Ecsv%2Egz
wget -nc -P "$RAW_DIR/tasic_2018" https://raw.githubusercontent.com/berenslab/rna-seq-tsne/master/data/tasic-sample_heatmap_plot_data.csv

# Hochgerner 2018
wget -nc -P "$RAW_DIR/hochgerner_2018" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104323/suppl/GSE104323_10X_expression_data_V2.tab.gz
wget -nc -P "$RAW_DIR/hochgerner_2018" ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104323/suppl/GSE104323_metadata_barcodes_24185cells.txt.gz


cat << EOF
================================================================================
Installing Python dependencies...
================================================================================
EOF

pip install -r requirements-figures.txt

cat << EOF
================================================================================
Preparing h5ad files...
================================================================================
EOF

mkdir -p "$H5AD_DIR"

echo "Generating macosko_2015.h5ad..."
time python -m generate_h5ad.macosko_2015 -i "$RAW_DIR/macosko_2015" -o "$H5AD_DIR"

echo "Generating shekhar_2016.h5ad..."
time python -m generate_h5ad.shekhar_2016 -i "$RAW_DIR/shekhar_2016" -o "$H5AD_DIR"

echo "Generating tasic_2018.h5ad..."
time python -m generate_h5ad.tasic_2018 -i "$RAW_DIR/tasic_2018" -o "$H5AD_DIR"

echo "Generating harris_2018.h5ad..."
time python -m generate_h5ad.harris_2018 -i "$RAW_DIR/harris_2018" -o "$H5AD_DIR"

echo "Generating hochgerner_2018.h5ad..."
time python -m generate_h5ad.hochgerner_2018 -i "$RAW_DIR/hochgerner_2018" -o "$H5AD_DIR"

echo "Generating cao_2019.h5ad..."
time python -m generate_h5ad.cao_2019 -i "$RAW_DIR/cao_2019" -o "$H5AD_DIR"
