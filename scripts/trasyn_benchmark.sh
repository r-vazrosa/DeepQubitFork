#!/bin/bash

mkdir -p tmp

for file in data/targets/1qubit/*.txt; do

    # converting text file format to npy format
    # and transposing the matrix
    NP_FILE=$(python -c "
import os
import numpy as np
from utils.matrix_utils import load_matrix_from_file
filename = '$file'
filename_base = os.path.basename(filename).split('.')[0]
_, U = load_matrix_from_file(filename)
out_filename = os.path.join('./tmp', filename_base + '.npy')
np.save(out_filename, U.T)
print(out_filename)
    ")

    # running trasyn synthesis
    echo -e -n "$file\t"
    OUTPUT=$({ time trasyn "$NP_FILE" 20 --error-threshold 0.07;  } 2>&1)

    # getting compilation time
    TIME=$(echo -e "$OUTPUT" | grep "real" | awk '/^real/ {print $2}')

    # getting info string
    INFO=$(echo -e "$OUTPUT" | grep "Sequence")

    echo -en "$TIME\t"
    echo -e "$INFO"
done
