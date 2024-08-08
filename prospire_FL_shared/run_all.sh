#!/bin/bash

# Run the script with 2 workers, 20 worker epochs and 100 epochs, save output to a file
./run_script.sh --n_workers 2 --w_epochs 20 --epochs 100 > output_2w_20se_100e.txt 2>&1

# Run the script with 3 workers, 20 worker epochs and 100 epochs, save output to a file
./run_script.sh --n_workers 3 --w_epochs 20 --epochs 100 > output_3w_20se_100e.txt 2>&1

# Run the script with 4 workers, 20 worker epochs and 100 epochs, save output to a file
./run_script.sh --n_workers 4 --w_epochs 20 --epochs 100 > output_4w_20se_100e.txt 2>&1
