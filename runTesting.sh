#!/bin/bash

# Convenience script to run perpetual random testing on HPC.
# Call this on a login node via screen and let it keep running until you have
# enough data.

# Module needed on Voltrino to use scikit-learn.
# module load python/3.6-anaconda-5.0.1

# Module needed on Eclipse to use scikit-learn.
module load anaconda3/2022.05

python3 testing.py
