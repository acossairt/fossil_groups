#!/bin/bash

# To run: open terminal
# make sure you're in correct conda env
# then execute:
# chmod +x run_fit_diffmah.sh
# ./run_fit_diffmah.sh

python fit_diffmah.py | tee fit_diffmah_output.txt