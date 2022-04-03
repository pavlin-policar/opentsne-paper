#!/bin/bash

set -e

H5AD_DIR="./data/h5ad"
FIGURES_DIR="./figs"


cat << EOF
================================================================================
Installing Python dependencies...
================================================================================
EOF

pip install -r requirements-figures.txt


cat << EOF
================================================================================
Generating figures...
================================================================================
EOF

mkdir -p "$FIGURES_DIR"

echo "Generating Figure 1..."
python fig1-macosko.py -i "$H5AD_DIR" -o "$FIGURES_DIR"
python fig2-tasic.py -i "$H5AD_DIR" -o "$FIGURES_DIR"
python fig3-cao.py -i "$H5AD_DIR" -o "$FIGURES_DIR"
python fig4-hochgerner.py -i "$H5AD_DIR" -o "$FIGURES_DIR"
