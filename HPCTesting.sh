#!/bin/bash

# Test all cases on short and long queues, limiting to max_jobs concurrent tasks.
# Testing occurs on already generated data from CSV files.

module load anaconda3/2022.05

mkdir ./output_$HOSTNAME

max_jobs=60

for model in {0..32}
do
    for app in LAMMPS ExaMiniMDsnap SWFFT nekbone HACC-IO
    do
        for baseline in 0 1
        do
            while [ $(squeue -u kmlamar | grep -c kmlamar) -ge $max_jobs ]
            do
                sleep 1
            done
            # Can add --depickle to the end of the Python command to generate
            # updated charts using models already made from previous runs, if
            # they exist (as pickle files).
            srun -N 1 -A fy140198 -J ProxyAppPredictor --partition=short,batch --time=4:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output_$HOSTNAME/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done