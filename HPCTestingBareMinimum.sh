#!/bin/bash

# Modified HPCTesting.sh script that only tests the results used in the
# paper's figures.

module load anaconda3/2022.05

mkdir ./output_$HOSTNAME

max_jobs=20

for model in {0..32}
do
    for app in ExaMiniMDsnap
    do
        for baseline in 0 1
        do
            while [ $(squeue -u kmlamar | grep -c kmlamar) -ge $max_jobs ]
            do
                sleep 1
            done
            srun -N 1 -A fy140198 -J ProxyAppPredictor --partition=short,batch --time=4:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output_$HOSTNAME/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done

for model in 0 12 19
do
    for app in SWFFT nekbone
    do
        for baseline in 0
        do
            while [ $(squeue -u kmlamar | grep -c kmlamar) -ge $max_jobs ]
            do
                sleep 1
            done
            srun -N 1 -A fy140198 -J ProxyAppPredictor --partition=short,batch --time=4:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output_$HOSTNAME/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done

for model in 18
do
    for app in LAMMPS SWFFT nekbone HACC-IO
    do
        for baseline in 0
        do
            while [ $(squeue -u kmlamar | grep -c kmlamar) -ge $max_jobs ]
            do
                sleep 1
            done
            srun -N 1 -A fy140198 -J ProxyAppPredictor --partition=short,batch --time=4:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output_$HOSTNAME/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done

for model in 19
do
    for app in LAMMPS HACC-IO
    do
        for baseline in 0
        do
            while [ $(squeue -u kmlamar | grep -c kmlamar) -ge $max_jobs ]
            do
                sleep 1
            done
            srun -N 1 -A fy140198 -J ProxyAppPredictor --partition=short,batch --time=4:00:00 python3 testing.py --doML --fromCSV --modelIdx $model --app $app --baseline $baseline > "./output_$HOSTNAME/${model}_${app}_${baseline}.txt" &
            sleep 1
        done
    done
done