"""A comprehensive script for generating, testing, and parsing inputs for a 
variety of Proxy Apps, both locally and on the Voltrino HPC testbed.
"""

import concurrent.futures
import copy
import multiprocessing
import math
import numbers
from re import X
import numpy as np
import os
import pandas as pd
import platform
import random
import signal
import subprocess
import time
import functools

from itertools import product
from pathlib import Path

from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# The default parameters for each application.
defaultParams = {}
# A set of sane parameter ranges.
# Our intent is to sweep through these with different input files.
rangeParams = {}
# A dictionary of Pandas DataFrames.
# Each application gets its own DataFrame.
df = {}
# This list will keep track of all active SLURM jobs.
# Will associate the app and index for tracking.
activeJobs = {}
# This dictionary will hold all inputs and outputs.
features = {}
# Used to identify what type of machine is being used.
SYSTEM = platform.node()
# Time to wait on SLURM, in seconds, to avoid a busy loop.
WAIT_TIME = 1
# Used to choose which apps to test.
# rangeParams.keys() #["ExaMiniMDbase", "ExaMiniMDsnap", "SWFFT", "sw4lite", "nekbone", "miniAMR"]
enabledApps = rangeParams.keys()
# Whether or not to shortcut out tests that may be redundant or invalid.
skipTests = True
# A terminate indicator. Set to True to quit gracefully.
terminate = False

snapFile = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
            '\n'
            '# Definition of SNAP potential Ta_Cand06A\n'
            '# Assumes 1 LAMMPS atom type\n'
            '\n'
            'variable zblcutinner equal 4\n'
            'variable zblcutouter equal 4.8\n'
            'variable zblz equal 73\n'
            '\n'
            '# Specify hybrid with SNAP, ZBL\n'
            '\n'
            'pair_style hybrid/overlay &\n'
            'zbl ${zblcutinner} ${zblcutouter} &\n'
            'snap\n'
            'pair_coeff 1 1 zbl ${zblz} ${zblz}\n'
            'pair_coeff * * snap Ta06A.snapcoeff Ta Ta06A.snapparam Ta\n')
snapcoeffFile = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
                 '\n'
                 '# LAMMPS SNAP coefficients for Ta_Cand06A\n'
                 '\n'
                 '1 31\n'
                 'Ta 0.5 1\n'
                 '-2.92477\n'
                 '-0.01137\n'
                 '-0.00775\n'
                 '-0.04907\n'
                 '-0.15047\n'
                 '0.09157\n'
                 '0.05590\n'
                 '0.05785\n'
                 '-0.11615\n'
                 '-0.17122\n'
                 '-0.10583\n'
                 '0.03941\n'
                 '-0.11284\n'
                 '0.03939\n'
                 '-0.07331\n'
                 '-0.06582\n'
                 '-0.09341\n'
                 '-0.10587\n'
                 '-0.15497\n'
                 '0.04820\n'
                 '0.00205\n'
                 '0.00060\n'
                 '-0.04898\n'
                 '-0.05084\n'
                 '-0.03371\n'
                 '-0.01441\n'
                 '-0.01501\n'
                 '-0.00599\n'
                 '-0.06373\n'
                 '0.03965\n'
                 '0.01072\n')
snapparamFile = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
                 '\n'
                 '# LAMMPS SNAP parameters for Ta_Cand06A\n'
                 '\n'
                 '# required\n'
                 'rcutfac 4.67637\n'
                 'twojmax 6\n'
                 '\n'
                 '# optional\n'
                 '\n'
                 'rfac0 0.99363\n'
                 'rmin0 0\n'
                 'diagonalstyle 3\n'
                 'bzeroflag 0\n'
                 'quadraticflag 0\n')

# A set of sane defaults based on 3d Lennard-Jones melt (in.lj).
defaultParams["ExaMiniMDbase"] = {"units": "lj",
                                  "lattice": "fcc",
                                  "lattice_constant": 0.8442,
                                  "lattice_offset_x": 0.0,
                                  "lattice_offset_y": 0.0,
                                  "lattice_offset_z": 0.0,
                                  "lattice_nx": 40,
                                  "lattice_ny": 40,
                                  "lattice_nz": 40,
                                  "ntypes": 1,
                                  "type": 1,
                                  "mass": 2.0,
                                  "force_type": "lj/cut",
                                  "force_cutoff": 2.5,
                                  "temperature_target": 1.4,
                                  "temperature_seed": 87287,
                                  "neighbor_skin": 0.3,
                                  "comm_exchange_rate": 20,
                                  "thermo_rate": 10,
                                  "dt": 0.005,
                                  "comm_newton": "off",
                                  "nsteps": 100,
                                  "nodes": 4,
                                  "tasks": 32}
rangeParams["ExaMiniMDbase"] = {"force_type": ["lj/cut"],
                                "lattice_nx": [1, 5, 40, 100, 200, 500],
                                "dt": [0.0001, 0.0005, 0.001, 0.005, 1.0, 2.0],
                                "nsteps": [0, 10, 100, 1000],
                                "nodes": [1, 4],
                                "tasks": [1, 32]}
# TODO: Fix the snap case for ExaMiniMD. Issues often occur when lattice_nx or dt are changed.
defaultParams["ExaMiniMDsnap"] = {"units": "metal",
                                  "lattice": "sc",
                                  "lattice_constant": 3.316,
                                  "lattice_offset_x": 0.0,
                                  "lattice_offset_y": 0.0,
                                  "lattice_offset_z": 0.0,
                                  "lattice_nx": 20,
                                  "lattice_ny": 20,
                                  "lattice_nz": 20,
                                  "ntypes": 1,
                                  "type": 1,
                                  "mass": 180.88,
                                  "force_type": "snap",
                                  "force_cutoff": None,
                                  "temperature_target": 300.0,
                                  "temperature_seed": 4928459,
                                  "neighbor_skin": 1.0,
                                  "comm_exchange_rate": 1,
                                  "thermo_rate": 5,
                                  "dt": 0.0005,
                                  "comm_newton": "on",
                                  "nsteps": 100,
                                  "nodes": 4,
                                  "tasks": 32}
rangeParams["ExaMiniMDsnap"] = {"force_type": ["snap"],
                                "lattice_nx": [1, 5, 40],
                                "dt": [0.0001, 0.0005, 0.001],
                                "nsteps": [0, 10, 100, 1000],
                                "nodes": [1, 4],
                                "tasks": [1, 32]}

defaultParams["SWFFT"] = {"n_repetitions": 1,
                          "ngx": 512,
                          "ngy": None,
                          "ngz": None,
                          "nodes": 4,
                          "tasks": 32}
rangeParams["SWFFT"] = {"n_repetitions": [1, 2, 4, 8, 16, 64],
                        "ngx": [256, 512, 1024, 2048],
                        "ngy": [None, 256, 512, 1024, 2048],
                        "ngz": [None, 256, 512, 1024, 2048],
                        "nodes": [1, 4],
                        "tasks": [1, 32]}

defaultParams["sw4lite"] = {"grid": True,
                            "gridny": None,
                            "gridnx": None,
                            "gridnz": None,
                            "gridx": 1.27,
                            "gridy": 1.27,
                            "gridz": 19.99,
                            "gridh": 0.003,
                            "developer": True,
                            "developercfl": None,
                            "developercheckfornan": 0,
                            "developerreporttiming": 1,
                            "developertrace": None,
                            "developerthblocki": None,
                            "developerthblockj": None,
                            "developerthblockk": None,
                            "developercorder": 0,
                            "topography": False,
                            "topographyzmax": 1.0,
                            "topographyorder": 6,
                            "topographyzetabreak": None,
                            "topographyinput": "gaussian",
                            "topographyfile": None,
                            "topographygaussianAmp": 0.1,
                            "topographygaussianXc": -.6,
                            "topographygaussianYc": 0.4,
                            "topographygaussianLx": 0.15,
                            "topographygaussianLy": 0.10,
                            "topographyanalyticalMetric": None,
                            "fileio": True,
                            "fileiopath": "gaussianHill-h0p01",
                            "fileioverbose": None,
                            "fileioprintcycle": None,
                            "fileiopfs": None,
                            "fileionwriters": None,
                            "time": True,
                            # Note: Separate time instantiations in example file.
                            "timet": 3.0,
                            "timesteps": 5,
                            "source": True,
                            "sourcem0": None,
                            "sourcex": 0.52,
                            "sourcey": 0.48,
                            "sourcez": None,
                            "sourcedepth": 0.2,
                            "sourceMxx": None,
                            "sourceMxy": 0.87,
                            "sourceMxz": None,
                            "sourceMyy": None,
                            "sourceMyz": None,
                            "sourceMzz": None,
                            "sourceFz": None,
                            "sourceFx": None,
                            "sourceFy": None,
                            "sourcet0": 0.36,
                            "sourcefreq": 16.6667,
                            "sourcef0": None,
                            "sourcetype": "Gaussian",
                            "supergrid": True,
                            "supergridgp": 30,
                            "supergriddc": None,
                            "testpointsource": False,
                            "testpointsourcecp": None,
                            "testpointsourcecs": None,
                            "testpointsourcerho": None,
                            "testpointsourcediractest": None,
                            "testpointsourcehalfspace": None,
                            "checkpoint": False,
                            "checkpointtime": None,
                            "checkpointtimeInterval": None,
                            "checkpointcycle": None,
                            "checkpointcycleInterval": None,
                            "checkpointfile": None,
                            "checkpointbufsize": None,
                            "restart": False,
                            "restartfile": None,
                            "restartbufsize": None,
                            "rec": True,
                            "recx": 0.7,
                            "recy": 0.6,
                            "reclat": None,
                            "reclon": None,
                            "recz": None,
                            "recdepth": 0,
                            "rectopodepth": None,
                            "recfile": "sta01",
                            "recsta": None,
                            "recnsew": None,
                            "recwriteEvery": None,
                            "recusgsformat": 1,
                            "recsacformat": 0,
                            "recvariables": None,
                            "block": True,
                            "blockrhograd": None,
                            "blockvpgrad": None,
                            "blockvsgrad": None,
                            "blockvp": 2.0,
                            "blockvs": 1.0,
                            "blockrho": None,
                            "blockr": 1.5,
                            "blockQs": None,
                            "blockQp": None,
                            "blockabsdepth": None,
                            "blockx1": None,
                            "blockx2": None,
                            "blocky1": None,
                            "blocky2": None,
                            "blockz1": None,
                            "blockz2": None,
                            "dgalerkin": None,
                            "dgalerkinorder": None,
                            "dgalerkinsinglemode": None,
                            "nodes": 4,
                            "tasks": 32}
rangeParams["sw4lite"] = {  # Done
                            "fileio": [False, True],
                            "fileiopath": [None],
                            "fileioverbose": [0, 5],
                            "fileioprintcycle": [1, 100],
                            "fileiopfs": [None],
                            "fileionwriters": [None],
                            # Done
                            "grid": [True],
                            "gridny": [2, 100],
                            "gridnx": [2, 100],
                            "gridnz": [2, 100],
                            "gridx": [1, 20],
                            "gridy": [1, 20],
                            "gridz": [1, 20],
                            "gridh": [0.001, 1.0],
                            # Done
                            "time": [True],
                            # Note: Multiple separate time instantiations in example file.
                            "timet": [1.0, 5.0],
                            "timesteps": [1, 10],
                            # Done
                            "supergrid": [True],
                            "supergridgp": [5, 30, 100],
                            "supergriddc": [0.01, 0.02, 0.05],
                            # Done
                            "source": [True],
                            "sourcem0": [0.0, 1.0],
                            "sourcex": [0.0, 1.0],
                            "sourcey": [0.0, 1.0],
                            "sourcez": [0.0, 1.0],
                            "sourcedepth": [0.0, 1.0],
                            "sourceMxx": [-1.0, 1.0],
                            "sourceMxy": [-1.0, 1.0],
                            "sourceMxz": [-1.0, 1.0],
                            "sourceMyy": [-1.0, 1.0],
                            "sourceMyz": [-1.0, 1.0],
                            "sourceMzz": [-1.0, 1.0],
                            "sourceFz": [-1.0, 1.0],
                            "sourceFx": [-1.0, 1.0],
                            "sourceFy": [-1.0, 1.0],
                            "sourcet0": [0.01, 0.50],
                            "sourcefreq": [0.1, 20.0],
                            "sourcef0": [0.0, 5.0],
                            "sourcetype": ["Ricker","Gaussian","Ramp","Triangle","Sawtooth","SmoothWave","Erf","GaussianInt","VerySmoothBump","RickerInt","Brune","BruneSmoothed","DBrune","GaussianWindow","Liu","Dirac","C6SmoothBump"],
                            # Done
                            "block": [False, True],
                            "blockrhograd": [0.1, 2.0],
                            "blockvpgrad": [0.1, 2.0],
                            "blockvsgrad": [0.1, 2.0],
                            "blockvp": [0.1, 2.0],
                            "blockvs": [0.1, 2.0],
                            "blockrho": [0.1, 2.0],
                            "blockr": [0.1, 2.0],
                            "blockQs": [None],
                            "blockQp": [None],
                            "blockabsdepth": [0, 1],
                            # For these, the valid values are between 0 and some max value specified by the grid size.
                            # Also, the 2s (max) must be larger than the 1s (min).
                            "blockx1": [None],
                            "blockx2": [None],
                            "blocky1": [None],
                            "blocky2": [None],
                            "blockz1": [None],
                            "blockz2": [None],
                            # Done
                            "topography": [False, True],
                            "topographyzmax": [0.0,1.0],
                            "topographyorder": [2,7],
                            "topographyzetabreak": [None],
                            "topographyinput": "gaussian",
                            "topographyfile": [None], # Unused
                            "topographygaussianAmp": [0.1,1.0],
                            "topographygaussianXc": [0.1,0.9],
                            "topographygaussianYc": [0.1,0.9],
                            "topographygaussianLx": [0.1,1.0],
                            "topographygaussianLy": [0.1,1.0],
                            "topographyanalyticalMetric": [None],
                            # Done
                            "rec": [False, True],
                            "recx": [None],
                            "recy": [None],
                            "reclat": [-90.0,90.0],
                            "reclon": [-180.0,180.0],
                            "recz": [None],
                            "recdepth": [None],
                            "rectopodepth": [None],
                            "recfile": [None],
                            "recsta": [None],
                            "recnsew": [0, 1],
                            "recwriteEvery": [100,10000],
                            "recusgsformat": [0,1],
                            "recsacformat": [0,1],
                            "recvariables": ["displacement","velocity","div","curl","strains","displacementgradient"],
                            # Done
                            # No plans to test this.
                            "checkpoint": [False],
                            "checkpointtime": [None],
                            "checkpointtimeInterval": [None],
                            "checkpointcycle": [None],
                            "checkpointcycleInterval": [None],
                            "checkpointfile": [None],
                            "checkpointbufsize": [None],
                            # Done
                            # No plans to test this.
                            "restart": [False],
                            "restartfile": [None],
                            "restartbufsize": [None],
                            # Done
                            # No plans to test this.
                            "dgalerkin": [False],
                            "dgalerkinorder": [None],
                            "dgalerkinsinglemode": [None],
                            # Done
                            # No plans to test this.
                            "developer": [False],
                            "developercfl": [None],
                            "developercheckfornan": [0],
                            "developerreporttiming": [1],
                            "developertrace": [None],
                            "developerthblocki": [None],
                            "developerthblockj": [None],
                            "developerthblockk": [None],
                            "developercorder": [0],
                            # Done
                            # No plans to test this.
                            "testpointsource": [False],
                            "testpointsourcecp": [None],
                            "testpointsourcecs": [None],
                            "testpointsourcerho": [None],
                            "testpointsourcediractest": [None],
                            "testpointsourcehalfspace": [None],
                            "nodes": [1, 4],
                            "tasks": [1, 32]}

defaultParams["nekbone"] = {"ifbrick": ".false.",
                            "iel0": 1,
                            "ielN": 50,
                            "istep": 1,
                            "nx0": 10,
                            "nxN": 10,
                            "nstep": 1,
                            "npx": 0,
                            "npy": 0,
                            "npz": 0,
                            "mx": 0,
                            "my": 0,
                            "mz": 0,
                            # nekbone is limited to 10 processors. It gets different rules.
                            "nodes": 1,
                            "tasks": 10}
rangeParams["nekbone"] = {"ifbrick": [".false.", ".true."],
                          "iel0": [1, 50],
                          "ielN": [50],
                          "istep": [1, 2],
                          "nx0": [2, 10],
                          "nxN": [10],
                          "nstep": [1, 2],
                          "npx": [0, 1, 10],
                          "npy": [0, 1, 10],
                          "npz": [0, 1, 10],
                          "mx": [0, 1, 10],
                          "my": [0, 1, 10],
                          "mz": [0, 1, 10],
                          "nodes": [1],
                          "tasks": [10]}

# NOTE: Several of these parameters are commented out as they are unsupported in the MPI version of the program.
defaultParams["miniAMR"] = {"--help": False,
                            "--nx": 10,
                            "--ny": 10,
                            "--nz": 10,
                            "--init_x": 1,
                            "--init_y": 1,
                            "--init_z": 1,
                            # NOTE: Default depends on load parameter. 1 is default for RCB.
                            "--reorder": 1,
                            # "load": "rcb",
                            "--npx": 1,
                            "--npy": 1,
                            "--npz": 1,
                            "--max_blocks": 500,
                            "--num_refine": 5,
                            "--block_change": 5,  # = num_refine.
                            "--uniform_refine": 1,
                            "--refine_freq": 5,
                            "--inbalance": 0,
                            "--lb_opt": 1,
                            "--num_vars": 40,
                            "--comm_vars": 0,
                            "--num_tsteps": 20,
                            "--time": None,
                            "--stages_per_ts": 20,
                            "--permute": False,
                            # "--blocking_send": False,
                            "--code": 0,
                            "--checksum_freq": 5,
                            "--stencil": 7,
                            "--error_tol": 8,
                            "--report_diffusion": False,
                            "--report_perf": 1,
                            # "--refine_ghosts": False,
                            # "--send_faces": False,
                            # "--change_dir": False,
                            # "--group_blocks": False,
                            # "--break_ties": False,
                            # "--limit_move": False,
                            "--num_objects": 0,
                            "type": 0,
                            "bounce": 0,
                            "center_x": 0.0,
                            "center_y": 0.0,
                            "center_z": 0.0,
                            "movement_x": 0.0,
                            "movement_y": 0.0,
                            "movement_z": 0.0,
                            "size_x": 1.0,
                            "size_y": 1.0,
                            "size_z": 1.0,
                            "inc_x": 0.0,
                            "inc_y": 0.0,
                            "inc_z": 0.0,
                            "nodes": 4,
                            "tasks": 32}
rangeParams["miniAMR"] = {"--help": [False],
                          "--nx": [10, 1000],
                          "--ny": [None],
                          "--nz": [None],
                          "--init_x": [1, 2, 4],
                          "--init_y": [None],
                          "--init_z": [None],
                          "--reorder": [0, 1],
                          # "load": ["rcb", "morton", "hilbert", "trunc_hilbert"],
                          "--npx": [1, 3],
                          "--npy": [None],
                          "--npz": [None],
                          "--max_blocks": [5, 500],
                          "--num_refine": [0, 5],
                          "--block_change": [0, 5],
                          "--uniform_refine": [0, 1],
                          # Ignored if uniform_refine=1
                          "--refine_freq": [1, 5],
                          "--inbalance": [0, 50],
                          "--lb_opt": [0, 1, 2],
                          "--num_vars": [1, 40],
                          "--comm_vars": [0, 40],
                          "--num_tsteps": [None, 20],
                          # Ignored if num_tsteps is used.
                          "--time": [None, 20],
                          "--stages_per_ts": [1, 20],
                          "--permute": [True, False],
                          # "--blocking_send": [True, False],
                          "--code": [0, 1, 2],
                          "--checksum_freq": [0, 5],
                          "--stencil": [0, 7, 27],
                          "--error_tol": [0, 8],
                          "--report_diffusion": [True, False],
                          # list(range(0,15+1)), # Can be set to 0 in tests to disable output.
                          "--report_perf": [0],
                          # "--refine_ghost": [True, False],
                          # "--send_faces": [True, False],
                          # "--change_dir": [True, False],
                          # "--group_blocks": [True, False],
                          # "--break_ties": [True, False],
                          # "--limit_move": [True, False],
                          "--num_objects": [0, 1],  # list(range(0,5)),
                          "type": list(range(0, 25+1)),
                          "bounce": [0, 1],
                          "center_x": [0.0, 1.0, -1.0],
                          "center_y": [None],
                          "center_z": [None],
                          "movement_x": [0.0, 1.0, -1.0],
                          "movement_y": [None],
                          "movement_z": [None],
                          "size_x": [0.0, 1.0, -1.0],
                          "size_y": [None],
                          "size_z": [None],
                          "inc_x": [0.0, 1.0, -1.0],
                          "inc_y": [None],
                          "inc_z": [None],
                          "nodes": [1, 4],
                          "tasks": [1, 32]}

# Convert the parameters list to a string.
# Used as comments on input files to make the parameters used clear.
# TODO: Remove/ignore this for deep learning.
def paramsToString(params):
    string = ""
    for param in params:
        string += param + "=" \
            + str("None" if params[param] == None else params[param]) + ","
    return string


# Get the next unused test index of the associated app.
# Enables extended testing.
def getNextIndex(app):
    try:
        idx = len(os.listdir("./tests/" + app + "/"))
    except FileNotFoundError:
        idx = 0
    return idx


def makeSLURMScript(f):
    # The base contents of the SLURM script.
    # Use format_map() to substitute parameters.
    contents = ('#!/bin/bash\n'
                '#SBATCH -N {nodes}\n'
                '#SBATCH --time=24:00:00\n'
                '#SBATCH -J {app}\n'
                'export OMP_NUM_THREADS=1\n'
                'export OMP_PLACES=threads\n'
                'export OMP_PROC_BIND=true\n'
                'echo "----------------------------------------"\n'
                'START_TS=$(date +"%s")\n'
                'echo "----------------------------------------"\n'
                'srun --ntasks-per-node={tasks} {command}\n'
                'echo "----------------------------------------"\n'
                'END_TS=$(date +"%s")\n'
                'DIFF=$(echo "$END_TS - $START_TS" | bc)\n'
                'echo "timeTaken = $DIFF"\n'
                'echo "----------------------------------------"').format_map(f)
    return contents


def makeFile(app, params):
    contents = ""
    if app.startswith("ExaMiniMD"):
        contents += "# " + paramsToString(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        if params["lattice_constant"] != None:
            contents += "lattice {lattice} {lattice_constant}\n".format_map(
                params)
        else:
            contents += "lattice {lattice} {lattice_offset_x} {lattice_offset_y} {lattice_offset_z}\n".format_map(
                params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(
            params)
        if params["ntypes"] != None:
            contents += "create_box {ntypes}\n".format_map(params)
        contents += "create_atoms\n"
        contents += "mass {type} {mass}\n".format_map(params)
        if params["force_type"] != "snap":
            contents += "pair_style {force_type} {force_cutoff}\n".format_map(
                params)
        else:
            contents += "pair_style {force_type}\n".format_map(params)
            contents += "pair_coeff * * Ta06A.snapcoeff Ta Ta06A.snapparam Ta\n"
        contents += "velocity all create {temperature_target} {temperature_seed}\n".format_map(
            params)
        contents += "neighbor {neighbor_skin}\n".format_map(params)
        contents += "neigh_modify every {comm_exchange_rate}\n".format_map(
            params)
        contents += "fix 1 all nve\n"
        contents += "thermo {thermo_rate}\n".format_map(params)
        contents += "timestep {dt}\n".format_map(params)
        contents += "newton {comm_newton}\n".format_map(params)
        contents += "run {nsteps}\n".format_map(params)
    elif app == "sw4lite":
        contents += "# " + paramsToString(params) + "\n\n"
        sections = ["fileio","grid","time","supergrid","source","block","topography","rec","checkpoint","restart","dgalerkin","developer","testpointsource"]
        for section in sections:
            if params[section]:
                contents += str(section) + " "
                for param in params:
                    if not param.startswith(section):
                        continue
                    if param == None:
                        continue
                    if param == section:
                        continue
                    contents += str(param.partition(section)[2]) + "=" + str(params[param]) + " "
                contents += "\n"
    elif app == "nekbone":
        contents = ('{ifbrick} = ifbrick ! brick or linear geometry\n'
                    '{iel0} {ielN} {istep} = iel0,ielN(per proc),stride ! range of number of elements per proc.\n'
                    '{nx0} {nxN} {nstep} = nx0,nxN,stride ! poly. order range for nx1\n'
                    '{npx} {npy} {npz} = npx,npy,npz ! np distrb, if != np, nekbone handle\n'
                    '{mx} {my} {mz} = mx,my,mz ! nelt distrb, if != nelt, nekbone handle\n').format_map(params)
    return contents


def getCommand(app, params):
    # Get the executable.
    # NOTE: Is there a better way than hardcoding these into the function?
    # Does it really matter anyway? I think not.
    if app.startswith("ExaMiniMD"):
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/ExaMiniMD/ExaMiniMD"
        else:
            exe = "../../../ExaMiniMD"
    elif app == "SWFFT":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/SWFFT/SWFFT"
        else:
            exe = "../../../SWFFT"
    elif app == "sw4lite":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/sw4lite/sw4lite"
        else:
            exe = "../../../sw4lite"
    elif app == "nekbone":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/nekbone/nekbone"
        else:
            exe = "../../../nekbone"
    elif app == "miniAMR":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/miniAMR/miniAMR.x"
        else:
            exe = "../../../miniAMR.x"

    args = ""
    if app.startswith("ExaMiniMD"):
        args = "-il input.lj"
    elif app == "SWFFT":
        # Locally adjust the params list to properly handle None.
        for param in params:
            params[param] = params[param] is not None and params[param] or ''
        args = "{n_repetitions} {ngx} {ngy} {ngz}".format_map(params)
    elif app == "sw4lite":
        args = "input.in"
    elif app == "nekbone":
        args = ""
    elif app == "miniAMR":
        for param in params:
            # Each of our standard parameters starts with "--".
            if param.startswith("--"):
                # If the parameter is unset, don't add it to the args list.
                if params[param] == False or params[param] == None or params[param] == '':
                    continue
                # If the parameter is a flag with no value, add it alone.
                if params[param] is True:
                    args += param + " "
                # Standard parameters add their name and value.
                else:
                    args += param + " " + str(params[param]) + " "
            # load is a special case.
            # Its value is the argument.
            if param == "load":
                args += "--" + params["load"] + " "
        # Create the number of objects we need to specify.
        for _ in range(params["--num_objects"]):
            # Fill in each of these arguments.
            args += "--object {type} {bounce} {center_x} {center_y} {center_z} {movement_x} {movement_y} {movement_z} {size_x} {size_y} {size_z} {inc_x} {inc_y} {inc_z} ".format_map(params)

    # Assemble the whole command.
    command = exe + " " + args
    return command


# Scrape the output for runtime, errors, etc.
def scrapeOutput(output, app, index):
    lines = output.split('\n')
    for line in lines:
        if line.startswith("timeTaken = "):
            features[app][index]["timeTaken"] = \
                int(line[len("timeTaken = "):])
        if "error" in line:
            # DEBUG: Ignore this error for now. Hopefully it isn't a problem.
            if "sbatch: error: spank: /opt/ovis/lib64/ovis-ldms/libjobinfo_slurm.so: Plugin file not found" in line:
                continue
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
        if "libhugetlbfs" in line:
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
    return features


def appendTest(app, test):
    global features

    # print("Appending test " + str(test) + " for app " + str(app))
    # The location of the output CSV.
    outputFile = "./tests/" + app + "dataset.csv"
    # We only add the header if we are creating the file fresh.
    needsHeader = not os.path.exists(outputFile)
    # Make sure the error key is there. Otherwise, it'll be missing sometimes.
    # NOTE: We need to do this for any field that is not guaranteed to be present in all tests. Additionally, things will break badly if new features are added in the future. We must start from scratch in such a case.
    if "error" not in features[app][test].keys():
        features[app][test]["error"] = ""
    # Convert the test to a DataFrame.
    dataframe = pd.DataFrame(features[app][test], index=[
                             features[app][test]["testNum"]])
    # Append to CSV.
    dataframe.to_csv(outputFile, mode='a', header=needsHeader)


def generateTest(app, prod, index):
    global activeJobs
    global features

    # These are the defaults right now.
    scriptParams = {"app": app,
                    "nodes": prod["nodes"],
                    "tasks": prod["tasks"]}

    # Get the default parameters, which we will adjust.
    params = copy.copy(defaultParams[app])
    # Update the params based on our cartesian product.
    params.update(prod)
    # Add the test number to the list of params.
    params["testNum"] = index

    # Initialize the app's dictionary.
    if app not in features:
        features[app] = {}
    # Add the parameters to a DataFrame.
    features[app][index] = params

    # Create a folder to hold a SLURM script and any input files needed.
    testPath = Path("./tests/" + app + "/" + str(index).zfill(10))
    testPath.mkdir(parents=True, exist_ok=True)

    # NOTE: Add support for compiling per-test binaries from here, if needed.

    # Generate the input file contents.
    fileString = makeFile(app, params)
    # If a fileString was generated
    if fileString != "":
        # Save the contents to an appropriately named file.
        if app.startswith("ExaMiniMD"):
            fileName = "input.lj"
        elif app == "sw4lite":
            fileName = "input.in"
        elif app == "nekbone":
            fileName = "data.rea"
        with open(testPath / fileName, "w+") as text_file:
            text_file.write(fileString)

    if app.startswith("ExaMiniMD") and params["force_type"] == "snap":
        # Copy in Ta06A.snap, Ta06A.snapcoeff, and Ta06A.snapparam.
        with open(testPath / "Ta06A.snap", "w+") as text_file:
            text_file.write(snapFile)
        with open(testPath / "Ta06A.snapcoeff", "w+") as text_file:
            text_file.write(snapcoeffFile)
        with open(testPath / "Ta06A.snapparam", "w+") as text_file:
            text_file.write(snapparamFile)

    # Get the full command, with executable and arguments.
    command = getCommand(app, params)
    # Set the command in the parameters.
    # Everything else was set earlier.
    scriptParams["command"] = command

    if SYSTEM == "voltrino-int":
        # Generate the SLURM script contents.
        SLURMString = makeSLURMScript(scriptParams)
        # Save the contents to an appropriately named file.
        with open(testPath / "submit.slurm", "w+") as text_file:
            text_file.write(SLURMString)

    # Wait until the test is ready to run.
    # On Voltrino, wait until the queue empties a bit.
    if SYSTEM == "voltrino-int":
        while True:
            # Get the number of jobs currently in my queue.
            nJobs = int(subprocess.run("squeue -u kmlamar | grep -c kmlamar", stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, shell=True, check=False, encoding='utf-8').stdout)
            # print("There are currently " + (str(nJobs) + " queued jobs for this user"))
            # If there is room on the queue, break out of the loop.
            # On my account, 5 jobs can run at once (MaxJobsPU),
            # 24 can be queued (MaxSubmitPU).
            # Can check by running: sacctmgr show qos format=MaxJobsPU,MaxSubmitPU
            if nJobs < 24:
                break
            # Wait before trying again.
            time.sleep(WAIT_TIME)
    # On local, do nothing.

    # Run the test case.
    # On Voltrino, submit the SLURM script.
    if SYSTEM == "voltrino-int":
        print("Queuing app: " + app + "\t test: " + str(index))
        output = subprocess.run("sbatch submit.slurm", cwd=testPath, shell=True, check=False,
                                encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        # If the output doesn't match, something went wrong.
        if not output.startswith("Submitted batch job "):
            print(output)
            return False
        # Add the queued job to our wait list.
        # We add a dictionary so we can keep track of things when we
        # handle the output later.
        activeJobs[int(output[len("Submitted batch job "):])] = \
            {"app": app, "index": index, "path": testPath}
    # On local, run the command.
    else:
        print("Running app: " + app + "\t test: " + str(index))
        start = time.time()
        output = str(subprocess.run(command, cwd=testPath, shell=True, check=False,
                     encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
        features[app][index]["timeTaken"] = time.time() - start
        # Save the command to file.
        with open(testPath / "command.txt", "w+") as text_file:
            text_file.write(command)
        # Save the output in the associated test's folder.
        with open(testPath / "output.txt", "w+") as text_file:
            text_file.write(output)
        features = scrapeOutput(output, app, index)
    return True


# Handle any unfinished outputs.
def finishJobs(lazy=False):
    global activeJobs
    global features

    # Keep running this loop until all active jobs have completed and been parsed.
    while len(activeJobs) > 0:
        # We want to finish a job each iteration.
        finishedAJob = False
        # Try to find a completed job in our active list.
        for job in activeJobs:
            # If the job is done, it will not be found in the queue.
            jobStatus = subprocess.run("squeue -j " + str(job),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False,
                                       encoding='utf-8').stdout
            # If the job is done.
            if "Invalid job id specified" in jobStatus \
                    or "kmlamar" not in jobStatus:
                # print("Parsing output from job: " + str(job)
                #       + "\tapp: " + activeJobs[job]["app"]
                #       + "\ttest: " + str(activeJobs[job]["index"]))
                # Open the file with the completed job.
                with open(activeJobs[job]["path"] / ("slurm-" + str(job) + ".out"), "r") as f:
                    output = f.read()
                # Parse the output.
                features = scrapeOutput(
                    output, activeJobs[job]["app"], activeJobs[job]["index"])
                # DEBUG: Report an error to screen.
                if(features[activeJobs[job]["app"]][activeJobs[job]["index"]].has_key("error")):
                    print(str(activeJobs[job]["app"]) + " " + str(activeJobs[job]["app"])+ " " + str(features[activeJobs[job]["app"]][activeJobs[job]["index"]]["error"]))
                # Save the output of this job to file.
                appendTest(activeJobs[job]["app"], activeJobs[job]["index"])
                # The job has been parsed. Remove it from the list.
                activeJobs.pop(job)
                # We successfully finished a job.
                finishedAJob = True
                # We have found our job in the for loop.
                # Break out and start the search again.
                break
        # If we went through the whole queue and all remaining jobs were still active.
        if not finishedAJob:
            # If we are lazily finishing jobs.
            if lazy:
                # Don't bother waiting. Break out now and come back later.
                break
            # Print the contents of the remaining queue.
            print(subprocess.run("squeue -u kmlamar", stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, shell=True, check=False, encoding='utf-8').stdout)
            # Wait before trying again.
            time.sleep(WAIT_TIME)


# This function tests the interactions between multiple parameters.
# It makes multiple adjustments to important parameters in the default file.
# The parameters enable resuming from an existing set of tests.
def adjustParams():
    global features
    global df

    # Loop through each Proxy App.
    for app in enabledApps:
        # Some apps are very particular.
        # Don't bother with them and stick to randomly generated tests only.
        if app == "sw4lite" or app == "miniAMR":
            continue
        # Identify where we left off, in case we already have some results.
        resumeIndex = getNextIndex(app)
        # Loop through each combination of parameter changes.
        # prod is the cartesian product of our adjusted parameters.
        for index, prod in enumerate((dict(zip(rangeParams[app], x)) for
                                      x in product(*rangeParams[app].values()))):
            # Skip iterations until we reach the target starting index.
            if resumeIndex > index:
                continue
            # Test skips and input hacks.
            if app.startswith("ExaMiniMD"):
                # A bit of a hack, but we don't really need to test all combinations of these.
                # Let lattice_nx dictate the values for lattice_ny and lattice_nz.
                prod["lattice_ny"] = prod["lattice_nx"]
                prod["lattice_nz"] = prod["lattice_nx"]
            elif app == "SWFFT":
                if skipTests:
                    if prod["ngy"] == None and prod["ngz"] != None:
                        # Skip this test. It is invalid.
                        print("Skipping invalid test " + str(index))
                        continue
            elif app == "nekbone":
                if skipTests:
                    skip = False
                    if prod["iel0"] > prod["ielN"]:
                        skip = True
                    if prod["nx0"] > prod["nxN"]:
                        skip = True
                    if skip:
                        print("Skipping invalid test " + str(index))
                        continue
            elif app == "miniAMR":
                prod["--ny"] = prod["--nx"]
                prod["--nz"] = prod["--nx"]
                prod["--init_y"] = prod["--init_x"]
                prod["--init_z"] = prod["--init_x"]
                prod["--npy"] = prod["--npx"]
                prod["--npz"] = prod["--npx"]
                prod["center_y"] = prod["center_x"]
                prod["center_z"] = prod["center_x"]
                prod["movement_y"] = prod["movement_x"]
                prod["movement_z"] = prod["movement_x"]
                prod["size_y"] = prod["size_x"]
                prod["size_z"] = prod["size_x"]
                prod["inc_y"] = prod["inc_x"]
                prod["inc_z"] = prod["inc_x"]

                # These cases are redundant because some parameters are ignored when others are set.
                if skipTests:
                    skip = False
                    if prod["--uniform_refine"] == 1 and prod["--refine_freq"] != 0:
                        skip = True
                    if prod["--num_tsteps"] != None and prod["--time"] != None:
                        #skip = True
                        if random.randint(0, 1):
                            prod["--num_tsteps"] = None
                        else:
                            prod["--time"] = None
                    # if prod["load"] == "hilbert" and prod["--reorder"] == 1:
                    #     skip = True
                    if prod["--max_blocks"] < (prod["--init_x"] * prod["--init_y"] * prod["--init_z"]):
                        prod["--max_blocks"] = prod["--init_x"] * \
                            prod["--init_y"] * prod["--init_z"]
                        skip = True
                    # if prod["load"] != "rcb" and prod["--change_dir"] == True:
                    #     skip = True
                    # if prod["load"] != "rcb" and prod["--break_ties"] == True:
                    #     skip = True

                    if skip:
                        print("Skipping redundant test " + str(index))
                        continue

            generateTest(app, prod, index)
            # Try to finish jobs part-way.
            finishJobs(lazy=True)

    finishJobs()

    # Legacy DataFrame conversion. Used to diagnose issues compared with my own attempts at writing to a CSV.
    # Convert each app dictionary to a DataFrame.
    for app in enabledApps:
        print("Saving DataFrame for app: " + app)
        df[app] = pd.DataFrame(features[app]).T
        # Save parameters and results to CSV for optional recovery.
        df[app].to_csv("./tests/" + app + "datasetClassic.csv")


# Read an existing DataFrame back from a saved CSV.
def readDF():
    global df

    # For each app.
    for app in enabledApps:
        # Open the existing CSV.
        df[app] = pd.read_csv("./tests/" + app + "dataset.csv",
                              sep=",", header=0, index_col=0, engine="c", quotechar="\"")
    return


def randParam(app, param, values=''):
    if values == '':
        values = rangeParams[app][param]
    # If it is a number:
    if isinstance(values[-1], numbers.Number):
        # Get the lowest value.
        minV = min(x for x in values if x is not None)
        # Get the highest value.
        maxV = max(x for x in values if x is not None)
        # Pick a random number between min and max to use as the parameter value.
        if isinstance(values[-1], float):
            return random.uniform(minV, maxV)
        elif isinstance(values[-1], int):
            return random.randint(minV, maxV)
        else:
            print("Found a range with type" + str(type(values[-1])))
            return random.randrange(minV, maxV)
    # Else if it has no meaningful range (ex. str):
    else:
        # Pick one of the values at random.
        return random.choice(values)

# Get a random, valid test case for the given app.
def getParams(app):
    params = {}
    # All jobs must set these.
    params["nodes"] = randParam(app, "nodes")
    params["tasks"] = randParam(app, "tasks")

    if app == "sw4lite":
        if random.choice(rangeParams[app]["fileio"]):
            params["fileio"] = True
            params["fileioverbose"] = randParam(app, "fileioverbose")
            params["fileioprintcycle"] = randParam(app, "fileioprintcycle")
        else:
            params["fileio"] = False

        params["grid"] = True
        def computeEndGridPoint(maxval, h):
            reltol = 1e-5
            abstol = 1e-12
            fnpts = round(maxval / h + 1)
            if math.fabs((fnpts - 1) * h - maxval) < reltol * math.fabs(maxval) + abstol:
                npts = int(fnpts)
            else:
                npts = int(fnpts) + 1
            return float(npts)
        if random.choice(range(2)):
            params["gridnx"] = randParam(app, "gridnx")
            params["gridny"] = randParam(app, "gridny")
            params["gridnz"] = randParam(app, "gridnz")
            params["gridh"] = randParam(app, "gridh")
            h = params["gridh"]
            xMax = (params["gridnx"]-1)*h
            yMax = (params["gridny"]-1)*h
            zMax = (params["gridnz"]-1)*h
        else:
            params["gridx"] = randParam(app, "gridx")
            params["gridy"] = randParam(app, "gridy")
            params["gridz"] = randParam(app, "gridz")
            choice = random.choice(range(4))
            if choice == 0:
                params["gridh"] = randParam(app, "gridh")
                h = params["gridh"]
                xMax = (computeEndGridPoint(params["gridx"], h)-1)*h
                yMax = (computeEndGridPoint(params["gridy"], h)-1)*h
                zMax = (computeEndGridPoint(params["gridz"], h)-1)*h
            elif choice == 1:
                params["gridnx"] = randParam(app, "gridnx")
                h = params["gridx"]/(params["gridnx"]-1)
                xMax = (params["gridnx"]-1)*h
                yMax = (computeEndGridPoint(params["gridy"], h)-1)*h
                zMax = (computeEndGridPoint(params["gridz"], h)-1)*h
            elif choice == 2:
                params["gridny"] = randParam(app, "gridny")
                h = params["gridy"]/(params["gridny"]-1)
                xMax = (computeEndGridPoint(params["gridx"], h)-1)*h
                yMax = (params["gridny"]-1)*h
                zMax = (computeEndGridPoint(params["gridz"], h)-1)*h
            elif choice == 3:
                params["gridnz"] = randParam(app, "gridnz")
                h = params["gridz"]/(params["gridnz"]-1)
                xMax = (computeEndGridPoint(params["gridx"], h)-1)*h
                yMax = (computeEndGridPoint(params["gridy"], h)-1)*h
                zMax = (params["gridnz"]-1)*h

        params["time"] = True
        if random.choice(range(2)):
            params["timet"] = randParam(app, "timet")
        else:
            params["timesteps"] = randParam(app, "timesteps")

        if random.choice(rangeParams[app]["supergrid"]):
            params["supergrid"] = True
            params["supergridgp"] = randParam(app, "supergridgp")
            params["supergriddc"] = randParam(app, "supergriddc")
        else:
            params["supergrid"] = False

        # TODO: Support more than 1 call of this.
        params["source"] = True
        choice = random.choice(range(2))
        if random.choice(range(2)):
            params["sourcex"] = randParam(app, "sourcex", [0.0, xMax])
            params["sourcey"] = randParam(app, "sourcey", [0.0, yMax])
        else:
            params["sourcex"] = randParam(app, "sourcex", [0.0, xMax])
            params["sourcey"] = randParam(app, "sourcey", [0.0, yMax])
        if random.choice(range(2)):
            params["sourcez"] = randParam(app, "sourcez", [0.0, zMax])
        else:
            params["sourcedepth"] = randParam(app, "sourcedepth", [0.0, zMax])
        if random.choice(range(2)):
            if random.choice(range(2)):
                params["sourceFx"] = randParam(app, "sourceFx")
            if random.choice(range(2)):
                params["sourceFy"] = randParam(app, "sourceFy")
            if random.choice(range(2)):
                params["sourceFz"] = randParam(app, "sourceFz")
            if "sourceFx" not in params and \
               "sourceFy" not in params and \
               "sourceFz" not in params:
                choice = random.choice(range(3))
                if choice == 0:
                    params["sourceFx"] = randParam(app, "sourceFx")
                elif choice == 1:
                    params["sourceFy"] = randParam(app, "sourceFy")
                elif choice == 2:
                    params["sourceFz"] = randParam(app, "sourceFz")
            if random.choice(range(2)):
                params["sourcef0"] = randParam(app, "sourcef0")
        else:
            if random.choice(range(2)):
                params["sourceMxx"] = randParam(app, "sourceMxx")
            if random.choice(range(2)):
                params["sourceMxy"] = randParam(app, "sourceMxy")
            if random.choice(range(2)):
                params["sourceMxz"] = randParam(app, "sourceMxz")
            if random.choice(range(2)):
                params["sourceMyy"] = randParam(app, "sourceMyy")
            if random.choice(range(2)):
                params["sourceMyz"] = randParam(app, "sourceMyz")
            if random.choice(range(2)):
                params["sourceMzz"] = randParam(app, "sourceMzz")
            if "sourceMxx" not in params and \
               "sourceMxy" not in params and \
               "sourceMxz" not in params and \
               "sourceMyy" not in params and \
               "sourceMyz" not in params and \
               "sourceMzz" not in params:
                choice = random.choice(range(6))
                if choice == 0:
                    params["sourceMxx"] = randParam(app, "sourceMxx")
                elif choice == 1:
                    params["sourceMxy"] = randParam(app, "sourceMxy")
                elif choice == 2:
                    params["sourceMxz"] = randParam(app, "sourceMxz")
                elif choice == 3:
                    params["sourceMyy"] = randParam(app, "sourceMyy")
                elif choice == 4:
                    params["sourceMyz"] = randParam(app, "sourceMyz")
                elif choice == 5:
                    params["sourceMzz"] = randParam(app, "sourceMzz")
            if random.choice(range(2)):
                params["sourcem0"] = randParam(app, "sourcem0")
        params["sourcetype"] = randParam(app, "sourcetype")
        params["sourcet0"] = randParam(app, "sourcet0")
        if params["sourcetype"] != "Dirac":
            params["sourcefreq"] = randParam(app, "sourcefreq")

        if random.choice(rangeParams[app]["topography"]):
            params["topography"] = True
            params["topographyinput"] = randParam(app, "topographyinput")
            params["topographyzmax"] = randParam(app, "topographyzmax")
            params["topographygaussianAmp"] = randParam(app, "topographygaussianAmp")
            params["topographygaussianXc"] = randParam(app, "topographygaussianXc")
            params["topographygaussianYc"] = randParam(app, "topographygaussianYc")
            params["topographygaussianLx"] = randParam(app, "topographygaussianLx")
            params["topographygaussianLy"] = randParam(app, "topographygaussianLy")
            params["topographyzetabreak"] = randParam(app, "topographyzetabreak")
            params["topographyorder"] = randParam(app, "topographyorder")
            params["topographyfile"] = randParam(app, "topographyfile")
            params["topographyanalyticalMetric"] = randParam(app, "topographyanalyticalMetric")
        else:
            params["topography"] = False

        if random.choice(rangeParams[app]["block"]):
            params["block"] = True
            params["blockvp"] = randParam(app, "blockvp")
            params["blockvs"] = randParam(app, "blockvs")
            if random.choice(range(2)):
                params["blockrho"] = randParam(app, "blockrho")
            else:
                params["blockr"] = randParam(app, "blockr")
            if random.choice(range(2)):
                params["blockx1"] = randParam(app, "blockx1", [0.0, xMax])
                params["blockx2"] = randParam(app, "blockx2", [0.0, xMax])
                if params["blockx1"] > params["blockx2"]:
                    # Swap
                    tmp = params["blockx1"]
                    params["blockx1"] = params["blockx2"]
                    params["blockx2"] = tmp
            if random.choice(range(2)):
                params["blocky1"] = randParam(app, "blocky1", [0.0, yMax])
                params["blocky2"] = randParam(app, "blocky2", [0.0, yMax])
                if params["blocky1"] > params["blocky2"]:
                    # Swap
                    tmp = params["blocky1"]
                    params["blocky1"] = params["blocky2"]
                    params["blocky2"] = tmp
            if random.choice(range(2)):
                params["blockz1"] = randParam(app, "blockz1", [0.0, zMax])
                params["blockz2"] = randParam(app, "blockz2", [0.0, zMax])
                if params["blockz1"] > params["blockz2"]:
                    # Swap
                    tmp = params["blockz1"]
                    params["blockz1"] = params["blockz2"]
                    params["blockz2"] = tmp
            if "topography" in params:
                params["blockabsdepth"] = randParam(app, "blockabsdepth")
            if random.choice(range(2)):
                params["blockrhograd"] = randParam(app, "blockrhograd")
            if random.choice(range(2)):
                params["blockvpgrad"] = randParam(app, "blockvpgrad")
            if random.choice(range(2)):
                params["blockvsgrad"] = randParam(app, "blockvsgrad")
        else:
            params["block"] = False

        if random.choice(rangeParams[app]["rec"]):
            params["rec"] = True
            if random.choice(range(2)):
                params["recx"] = randParam(app, "recx", [0.0, xMax])
                params["recy"] = randParam(app, "recy", [0.0, yMax])
            else:
                params["reclat"] = randParam(app, "reclat")
                params["reclon"] = randParam(app, "reclon")
            choice = random.choice(range(3))
            if choice == 0:
                params["recz"] = randParam(app, "recz", [0.0, zMax])
            elif choice == 1:
                params["recdepth"] = randParam(app, "recdepth", [0.0, zMax])
            elif choice == 2:
                params["rectopodepth"] = randParam(app, "rectopodepth", [0.0, zMax])
            params["recfile"] = randParam(app, "recfile")
            params["recsta"] = randParam(app, "recsta")
            params["recnsew"] = randParam(app, "recnsew")
            params["recwriteEvery"] = randParam(app, "recwriteEvery")
            if random.choice(range(2)):
                params["recusgsformat"] = 1
                params["recsacformat"] = 0
            else:
                params["recusgsformat"] = 0
                params["recsacformat"] = 1
            params["recvariables"] = randParam(app, "recvariables")
        else:
            params["rec"] = False

    elif app == "SWFFT":
        params["n_repetitions"] = randParam(app, "n_repetitions")

        # Confirm the number is a power of 2.
        def isPow2(x):
            return (x & (x-1) == 0) and x != 0
        # Round up to the nearest power of 2.
        def nextPow2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()
        if random.choice(range(2)):
            params["ngx"] = randParam(app, "ngx")
            if not isPow2(params["ngx"]):
                params["ngx"] = nextPow2(params["ngx"])
        else:
            params["ngx"] = None
        if params["ngx"] is None and random.choice(range(2)):
            params["ngy"] = randParam(app, "ngy")
            if not isPow2(params["ngy"]):
                params["ngy"] = nextPow2(params["ngy"])
        else:
            params["ngy"] = None
        if params["ngy"] is None and random.choice(range(2)):
            params["ngz"] = randParam(app, "ngz")
            if not isPow2(params["ngz"]):
                params["ngz"] = nextPow2(params["ngz"])
        else:
            params["ngz"] = None

    elif app == "miniAMR":
        params["--help"] = None

        params["--nx"] = randParam(app, "--nx")
        if params["--nx"] % 2 != 0:
            params["--nx"] += 1
        params["--ny"] = randParam(app, "--nx")
        if params["--ny"] % 2 != 0:
            params["--ny"] += 1
        params["--nz"] = randParam(app, "--nx")
        if params["--nz"] % 2 != 0:
            params["--nz"] += 1

        params["--init_x"] = randParam(app, "--init_x")
        params["--init_y"] = randParam(app, "--init_x")
        params["--init_z"] = randParam(app, "--init_x")

        params["--reorder"] = randParam(app, "--reorder")

        def factors(n):
            return set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
        procCount = []
        procCount.append(random.choice(list(factors(multiprocessing.cpu_count()))))
        procCount.append(random.choice(list(factors(multiprocessing.cpu_count()/procCount[0]))))
        procCount.append(int(multiprocessing.cpu_count()/procCount[0]/procCount[1]))
        random.shuffle(procCount)
        params["--npx"] = procCount.pop()
        params["--npy"] = procCount.pop()
        params["--npz"] = procCount.pop()

        params["--max_blocks"] = randParam(app, "--max_blocks")
        if params["--max_blocks"] < params["--init_x"] * params["--init_y"] * params["--init_z"]:
            params["--max_blocks"] = params["--init_x"] * params["--init_y"] * params["--init_z"]

        params["--num_refine"] = randParam(app, "--num_refine")
        params["--block_change"] = randParam(app, "--block_change")
        
        params["--uniform_refine"] = randParam(app, "--uniform_refine")
        if params["--uniform_refine"] != 1:
            params["--refine_freq"] = randParam(app, "--refine_freq")
        else:
            params["--refine_freq"] = None
        
        params["--inbalance"] = randParam(app, "--inbalance")
        params["--lb_opt"] = randParam(app, "--lb_opt")

        params["--num_vars"] = randParam(app, "--num_vars")
        params["--comm_vars"] = randParam(app, "--comm_vars", [0, params["--num_vars"]])

        if random.choice(range(2)):
            params["--num_tsteps"] = randParam(app, "--num_tsteps")
        else:
            params["--time"] = randParam(app, "--time")

        params["--stages_per_ts"] = randParam(app, "--stages_per_ts")
        params["--permute"] = randParam(app, "--permute")
        params["--code"] = randParam(app, "--code")
        params["--checksum_freq"] = randParam(app, "--checksum_freq")
        params["--stencil"] = random.choice(rangeParams["miniAMR"]["--stencil"])
        params["--error_tol"] = randParam(app, "--error_tol")
        params["--report_diffusion"] = randParam(app, "--report_diffusion")
        params["--report_perf"] = randParam(app, "--report_perf")

        params["--num_objects"] = randParam(app, "--num_objects")
        if params["--num_objects"] > 0:
            params["type"] = randParam(app, "type")
            params["bounce"] = randParam(app, "bounce")
            params["center_x"] = randParam(app, "center_x")
            params["center_y"] = randParam(app, "center_x")
            params["center_z"] = randParam(app, "center_x")
            params["movement_x"] = randParam(app, "movement_x")
            params["movement_y"] = randParam(app, "movement_x")
            params["movement_z"] = randParam(app, "movement_x")
            params["size_x"] = randParam(app, "size_x")
            params["size_y"] = randParam(app, "size_x")
            params["size_z"] = randParam(app, "size_x")
            params["inc_x"] = randParam(app, "inc_x")
            params["inc_y"] = randParam(app, "inc_x")
            params["inc_z"] = randParam(app, "inc_x")

    # TODO: Handle conditional cases for each app here.
    # elif app == "ExaMiniMDbase":
    #     pass
    # elif app = "ExaMiniMDsnap":
    #     pass
    # elif app == "nekbone":
    #     pass

    # The default case just picks parameters at random within a range.
    else:
        # For each parameter:
        for param, values in rangeParams[app].items():
            params[param] = randParam(app, param)
    
    return params


# Run random permutations of tests outside of the specific set of tests we have defined. This extra variety helps training.
def randomTests():
    global terminate
    signal.signal(signal.SIGINT, exit_gracefully)
    # Cancel via Ctrl+C.
    # While we have not canceled the test:
    while not terminate:
        # Pick a random app.
        app = random.choice(list(enabledApps))
        # Get the index to save the test files.
        index = getNextIndex(app)
        # Get the parameters.
        params = getParams(app)
        # Run the test.
        generateTest(app, params, index)
        # Try to finish jobs.
        finishJobs(lazy=True)
        # If we want to terminate, we can't be lazy. Be sure all jobs complete.
    finishJobs(lazy=False)
    signal.signal(signal.SIGINT, original_sigint)

# Train and test a regressor on a dataset.
def regression(regressor, modelName, X, y):
    ret = str(modelName) + "\n"
    # Train Regressor.
    regressor = regressor.fit(X, y)
    # Run and report cross-validation accuracy.
    scores = cross_val_score(regressor, X, y, cv=5,
                             scoring="r2")
    ret += " R^2: " + str(scores.mean()) + "\n"
    scores = cross_val_score(regressor, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    ret += " RMSE: " + str(scores.mean()) + "\n"

    # Retrain on 4/5 of the data for plotting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    regressor = regressor.fit(X_train, y_train)
    # Plot the results for each regressor.
    y_pred = regressor.predict(X_test)
    plt.figure()
    plt.scatter(y_pred, y_test, s=20, c="black", label="data")
    plt.xlabel("Predicted ("+str(modelName)+")")
    plt.ylabel("Actual")
    plt.legend()
    plt.savefig("figures/"+str(modelName).replace(" ", "")+".svg")

    print(X.columns)
    print(regressor.feature_importances_)
    tree.plot_tree(regressor)
    plt.show()

    print(str(modelName))
    return ret


def runRegressors(X, y, app=""):
    # Make sure our features have the expected shape.
    # Also useful to keep track of test sizes.
    ret = str(X.shape) + "\n"

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = []

        # Run our regressors.
        # forestRegressor = RandomForestRegressor()
        # futures.append(executor.submit(
        #     regression, forestRegressor, "Random Forest Regressor "+app, X, y))

        # futures.append(executor.submit(
        #     regression, linear_model.BayesianRidge(), "Bayesian Ridge "+app, X, y))

        # futures.append(executor.submit(regression, svm.SVR(),
        #                "Support Vector Regression RBF "+app, X, y))
        # for i in range(1, 3+1):
        #     futures.append(executor.submit(regression, svm.SVR(
        #         kernel="poly", degree=i), "Support Vector Regression poly "+str(i)+" "+app, X, y))
        # futures.append(executor.submit(regression, svm.SVR(
        #     kernel="sigmoid"), "Support Vector Regression sigmoid "+app, X, y))

        # futures.append(executor.submit(regression, make_pipeline(StandardScaler(), SGDRegressor(
        #     max_iter=1000, tol=1e-3)), "Linear Stochastic Gradient Descent Regressor "+app, X, y))

        # for i in range(1, 7+1):
        #     futures.append(executor.submit(regression, KNeighborsRegressor(
        #         n_neighbors=i), str(i)+" Nearest Neighbors Regressor "+app, X, y))

        # for i in range(1, 4+1):
        #     futures.append(executor.submit(regression, PLSRegression(
        #         n_components=i), str(i)+" PLS Regression "+app, X, y))

        futures.append(executor.submit(
            regression, tree.DecisionTreeRegressor(), "Decision Tree Regressor "+app, X, y))

        # for i in range(1, 10+1):
        #     layers = tuple(100 for _ in range(i))
        #     futures.append(executor.submit(regression, MLPRegressor(
        #         activation="relu", hidden_layer_sizes=layers, random_state=1, max_iter=500), str(i)+" MLP Regressor relu "+app, X, y))

        for future in futures:
            ret += future.result()
    return ret


# Run machine learning on the DataFrames.
def ML():
    global df

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = []
        for app in enabledApps:
            # Print the app name, so we keep track of which one is being worked on.
            print("\n" + app)
            futures.append(executor.submit(str, "\n" + app + "\n"))
            X = df[app]

            # Use the error field to report simply whether or not we encountered an
            # error. We can use this as a training feature.
            if "error" in X.columns:
                X["error"] = X["error"].notnull()
                # Filter out errors.
                # X = X[X["error"].isnull()]
                # X = X.drop(columns="error")
                # if X.shape[0] < 1:
                #     print("All tests contained errors. Skipping...")
                #     continue
            # Simple replacements for base cases.
            X = X.replace('on', '1', regex=True)
            X = X.replace('off', '0', regex=True)
            X = X.replace('true', '1', regex=True)
            X = X.replace('false', '0', regex=True)
            X = X.replace('.true.', '1', regex=True)
            X = X.replace('.false.', '0', regex=True)
            # Convert non-numerics via an encoder.
            le = preprocessing.LabelEncoder()
            # Iterate over every cell.
            for col in X:
                for rowIndex, row in X[col].iteritems():
                    try:
                        # If it can be a float, make it a float.
                        X[col][rowIndex] = float(X[col][rowIndex])
                        # If the float is NaN (unacceptable to Sci-kit), make it -1.0 for now.
                        if pd.isna(X[col][rowIndex]):
                            X[col][rowIndex] = -1.0
                    except ValueError:
                        # Otherwise, use an encoder to make it numeric.
                        X[col] = X[col].astype(str)
                        X[col] = le.fit_transform(X[col])
            y = X["error"].astype(float)
            # The time taken should be an output, not an input.
            # y = X["timeTaken"].astype(float)
            X = X.drop(columns="timeTaken")
            # When predicting, we cannot know if the program crashed before it starts.
            X = X.drop(columns="error")
            # The testNum is also irrelevant for training purposes.
            X = X.drop(columns="testNum")

            if app == "ExaMiniMD":
                X = X.drop(columns="units")
                X = X.drop(columns="lattice")
                X = X.drop(columns="lattice_constant")
                X = X.drop(columns="lattice_offset_x")
                X = X.drop(columns="lattice_offset_y")
                X = X.drop(columns="lattice_offset_z")
                X = X.drop(columns="lattice_ny")
                X = X.drop(columns="lattice_nz")
                X = X.drop(columns="ntypes")
                X = X.drop(columns="type")
                X = X.drop(columns="mass")
                X = X.drop(columns="force_cutoff")
                X = X.drop(columns="temperature_target")
                X = X.drop(columns="temperature_seed")
                X = X.drop(columns="neighbor_skin")
                X = X.drop(columns="comm_exchange_rate")
                X = X.drop(columns="thermo_rate")
                X = X.drop(columns="comm_newton")

            # # Standardization.
            # scaler = preprocessing.StandardScaler().fit(X)
            # X = scaler.transform(X)
            # # Feature selection. Removes useless columns to simplify the model.
            # sel = feature_selection.VarianceThreshold(threshold=0)
            # X = sel.fit_transform(X)
            # # Discretization. Buckets results to whole minutes like related works.
            # # y = y.apply(lambda x: int(x/60))

            # Run regressors.
            futures.append(executor.submit(runRegressors, X, y, app+" base"))

        print('Writing output. Waiting for tests to complete.')
        with open('MLoutput.txt', 'w') as f:
            for future in futures:
                result = future.result()
                f.write(str(result))
                print(result)

# Used to run a small set of hardcoded test cases.
# Useful for debugging purposes.
def baseTest():
    app = "ExaMiniMDsnap"

    params = copy.copy(defaultParams[app])
    params["lattice_nx"] = 1
    generateTest(app, params, getNextIndex(app))

    params = copy.copy(defaultParams[app])
    params["lattice_nx"] = 200
    generateTest(app, params, getNextIndex(app))

    params = copy.copy(defaultParams[app])
    params["dt"] = 0.0001
    generateTest(app, params, getNextIndex(app))

    params = copy.copy(defaultParams[app])
    params["dt"] = 2.0
    generateTest(app, params, getNextIndex(app))

    finishJobs()


def main():
    # baseTest()
    # return
    fromCSV = False

    # Optionally start training from CSV immediately.
    if fromCSV:
        readDF()
    else:
        # Run through all of the primary tests.
        # adjustParams()
        # Run tests at random indefinitely.
        randomTests()
    # Perform machine learning.
    # ML()


def exit_gracefully(signum, frame):
    global terminate
    # Restore the original signal handler as otherwise bad things will happen
    # in input when CTRL+C is pressed, and our signal handler is not re-entrant.
    signal.signal(signal.SIGINT, original_sigint)
    try:
        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            terminate = True
    except KeyboardInterrupt:
        print("Ok ok, quitting.")
        terminate = True
    # Restore the exit gracefully handler here.
    signal.signal(signal.SIGINT, exit_gracefully)


if __name__ == "__main__":
    # store the original SIGINT handler.
    original_sigint = signal.getsignal(signal.SIGINT)
    # Run main.
    main()
