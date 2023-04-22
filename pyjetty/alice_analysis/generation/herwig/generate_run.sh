#!/bin/bash

#BASE_DIR=/home/james/pyjetty/pyjetty/alice_analysis/generatioa/herwig
BASE_DIR=/home/software/users/wenqing/pyjetty/pyjetty/alice_analysis/generation/herwig

for BIN in $(seq 5 6);
do
    echo "Generating bin: $BIN"
    cd $BASE_DIR/run/$BIN
    Herwig read $BASE_DIR/config/$BIN/LHC_5020_MPI.in
done
