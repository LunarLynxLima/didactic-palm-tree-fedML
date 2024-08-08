#!/bin/bash

# Default values for arguments
n_workers=1
w_epochs=0
epochs=200

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_workers) n_workers="$2"; shift ;;
        --w_epochs) w_epochs="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the provided arguments
python3 prospire_fedml.py --n_workers $n_workers --w_epochs $w_epochs --epochs $epochs
