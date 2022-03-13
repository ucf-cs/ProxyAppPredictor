#!/bin/bash
## Author: Omar Aaziz
## Date: 01/19/2022
## This script initialize the ldmsd samplers and aggregator instances on the running nodes
## Run this script before the application run line to initiate ldms to collect data
##
## Input:
##    ldms_out_path, path where ldms run files and store output will be saved
##    TODO: (OPTIONAL) A json file that specifies the sampler plugins to load, if no file supplied then a default sampler will be loaded


if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Usage: ldms_init.sh <ldms_out_path>"
fi

## topPath, is the path where the ldms data would be saved
export topPath=$1
export job_ldms_dir=$topPath/ldms_$SLURM_JOBID
## LDMS_PREFIX, the ldms build folder
export LDMS_PREFIX=/projects/HPCMON/INSTALL/ovis_voltrino_master
## LDMSD_LOG_LEVEL, the logging mode (INFO,DEBUG)
export LDMSD_LOG_LEVEL='DEBUG'
#export LDMSD_LOG_LEVEL='INFO'
export OVIS_HOME=$LDMS_PREFIX

echo "############### LDMS START ######################"
## Run LDMS script
srun --exclusive -m cyclic --ntasks-per-node=1 --label /home/kmlamar/ProxyAppPredictor/start_ldms_srun.sh $SLURM_JOB_ID $job_ldms_dir $HOME/.ldms_slurm_env.sh &
echo '------------------------------'
sleep 25 ; # give ldms time to start and connect up
# NOTE: Sleep also (in theory) provides some up-front data to train on for predictions.
# Using this as training data likely won't do much for us until we're deployed on a production system.
echo '------------------------------'
echo '$PATH'
echo '------------------------------'
echo '$HOSTNAME'
echo '------------------------------'
