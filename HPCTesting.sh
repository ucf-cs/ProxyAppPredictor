#!/bin/bash

module load anaconda3/2022.05

mkdir ./output

for model in {0..32}
do
    for app in LAMMPS ExaMiniMDsnap SWFFT nekbone HACC-IO
    do
        for baseline in 0 1
        do
            srun -N 1 -A fy140198 -J ProxyAppPredictor --time=24:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done