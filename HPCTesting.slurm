#!/bin/bash
#SBATCH -N 1
#SBATCH -J ProxyAppPredictor
#SBATCH -A fy140198

module load anaconda3/2022.05

mkdir ./output

for model in {0..32}
do
    for app in LAMMPS ExaMiniMDsnap SWFFT nekbone HACC-IO
    do
        for baseline in 0 1
        do
            python3 testing.py True --modelIdx $model --app $app --baseline $baseline > "./output/${model}_${app}_${baseline}.txt"
        done
    done
done