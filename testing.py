"""A comprehensive script for generating, testing, and parsing inputs for a
variety of Proxy Apps, both locally and on the Voltrino HPC testbed.
"""

import argparse
from cmath import inf, nan
import concurrent.futures
import copy
import multiprocessing
import math
import numbers
from re import X
import numpy as np
import os
import pandas as pd
import pickle
import platform
import random
import signal
import subprocess
import time
import functools

from itertools import product
from itertools import count
from pathlib import Path

from predictors.analytical.analytical_regressor import AnalyticalRegressor
from predictors.Tsafrir.tsafrir_regressor import TsafrirRegressor
from predictors.reqtime.reqtime_regressor import ReqtimeRegressor
from predictors.complete.complete_regressor import CompleteRegressor

# These cannot be used on Voltrino.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
# from complete.complete_regressor import CompleteRegressor
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# The default parameters for each application.
default_params = {}
# A set of sane parameter ranges.
# Our intent is to sweep through these with different input files.
range_params = {}
# A dictionary of Pandas DataFrames.
# Each application gets its own DataFrame.
df = {}

# This list will keep track of all jobs yet to be queued.
queued_jobs = {}
# This list will keep track of all active SLURM jobs.
# Will associate the app and index for tracking.
active_jobs = {}
# This dictionary will hold all inputs and outputs.
features = {}

# Used to identify what type of machine is being used.
SYSTEM = platform.node()
# The location where all test output is saved.
# TEST_DIR = "/pscratch/kmlamar/ProxyAppPredictorTests" + SYSTEM + "/" #"./tests/"
TEST_DIR = "./tests/"
# Where ML results are pickled and stored.
PICKLE_DIR = "/pscratch/kmlamar/pickles/"
# Time to wait on SLURM, in seconds, to avoid a busy loop.
WAIT_TIME = 1
# The maximum number of active jobs to enqueue.
# Up to 24 can be queued on Voltrino (MaxSubmitPU).
# Up to 50 is reasonable for larger systems.
MAX_JOBS = 50
# The maximum number of internally queued jobs to enqueue.
# The larger this number, the greater the typical distance between tests with
# the same inputs.
MAX_QUEUE = 30
# The number of times to run the same test.
# Used to measure run time variability.
REPEAT_COUNT = 5
# Runs apps with their debug symbols enabled.
# Set to false for performance testing.
# NOTE: No debug apps available on eclipse at this time.
DEBUG_APPS = False
# Used to choose which apps to test.
# range_params.keys() #["ExaMiniMDbase", "ExaMiniMDsnap", "SWFFT", "sw4lite", "nekbone", "miniAMR", "HACC-IO", "LAMMPS"]
# NOTE: These are the apps I have in my draft.
enabled_apps = ["ExaMiniMDsnap",  "HACC-IO", "LAMMPS", "SWFFT", "nekbone"]
# Whether or not to shortcut out tests that may be redundant or invalid.
SKIP_TESTS = True
# A global terminate indicator. Set to True to quit gracefully.
terminate = False

should_depickle = False

# LAMMPS SNAP file contents for ExaMiniMD SNAP testing.
SNAP_FILE = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
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
SNAPCOEFF_FILE = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
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
SNAPPARAM_FILE = ('# DATE: 2014-09-05 CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)\n'
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
default_params["ExaMiniMDbase"] = {"units": "lj",
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
range_params["ExaMiniMDbase"] = {"force_type": ["lj/cut"],
                                 "lattice_nx": [1, 5, 26],
                                 "lattice_ny": [1, 5, 26],
                                 "lattice_nz": [1, 5, 26],
                                 "dt": [0.0001, 0.0005, 0.001, 0.005, 1.0, 2.0],
                                 "nsteps": [0, 10, 100, 1000],
                                 "nodes": [1, 64],
                                 "tasks": [1, 32]}
default_params["ExaMiniMDsnap"] = {"units": "metal",
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
range_params["ExaMiniMDsnap"] = {"force_type": ["snap"],
                                 "lattice_nx": [1, 5, 26],
                                 "lattice_ny": [1, 5, 26],
                                 "lattice_nz": [1, 5, 26],
                                 "dt": [0.0001, 0.0005, 0.001],
                                 "nsteps": [0, 10, 100, 1000, 10000],
                                 "nodes": [1, 64],
                                 "tasks": [1, 32]}

default_params["LAMMPS"] = {"units": "lj",
                            "lattice": "fcc",
                            "lattice_constant": 0.8842,  # rhostar
                            "lattice_nx": 5,
                            "lattice_ny": 5,
                            "lattice_nz": 5,
                            "ntypes": 1,
                            "type": 1,
                            "mass": 1.0,
                            "temperature_target": 1.0,  # tinitial
                            "temperature_seed": 12345,
                            "force_type": "lj/cut",
                            "force_cutoff": 2.5,  # cutoff
                            "neighbor_skin": 0.3,
                            "comm_exchange_rate": 20,
                            "dt": 0.005,
                            "thermo_rate": 10,
                            "nsteps": 100,
                            "nodes": 4,
                            "tasks": 32}
range_params["LAMMPS"] = {"lattice_nx": [1, 5, 26],
                          "lattice_ny": [1, 5, 26],
                          "lattice_nz": [1, 5, 26],
                          "lattice_constant": [0.5, 0.8842, 1.0],
                          "dt": [0.0001, 0.0005, 0.001],
                          "force_cutoff": [2.0, 2.5, 3.0],
                          "neighbor_skin": [0.1, 0.3, 0.5],
                          "temperature_target": [1.0],
                          "thermo_rate": [1, 10, 100],
                          "nsteps": [0, 10, 100, 1000, 10000],
                          "nodes": [1, 64],
                          "tasks": [1, 32]}

default_params["SWFFT"] = {"n_repetitions": 1,
                           "ngx": 512,
                           "ngy": None,
                           "ngz": None,
                           "nodes": 4,
                           "tasks": 32}
range_params["SWFFT"] = {"n_repetitions": [1, 2, 4, 8, 16, 64, 512],
                         "ngx": [256, 512, 1024],
                         "ngy": [None, 256, 512, 1024],
                         "ngz": [None, 256, 512, 1024],
                         "nodes": [1, 64],
                         "tasks": [1, 31]}

default_params["sw4lite"] = {"grid": True,
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
                             # Note: Multiple time instantiations in example file.
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
range_params["sw4lite"] = {
    "fileio": [False, True],
    "fileiopath": [None],
    "fileioverbose": [0, 5],
    "fileioprintcycle": [1, 100],
    "fileiopfs": [None],
    "fileionwriters": [None],

    "grid": [True],
    "gridny": [0.001, 1.0],
    "gridnx": [0.001, 1.0],
    "gridnz": [0.001, 1.0],
    "gridx": [0.001, 1.0],
    "gridy": [0.001, 1.0],
    "gridz": [0.001, 1.0],
    "gridh": [0.001, 1.0],

    "time": [True],
    "timet": [0.1, 1.0],
    "timesteps": [1, 5],

    "supergrid": [True],
    "supergridgp": [5, 30],
    "supergriddc": [0.01, 0.02, 0.05],

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
    "sourcetype": ["Ricker", "Gaussian", "Ramp", "Triangle", "Sawtooth", "SmoothWave", "Erf", "GaussianInt", "VerySmoothBump", "RickerInt", "Brune", "BruneSmoothed", "DBrune", "GaussianWindow", "Liu", "Dirac", "C6SmoothBump"],

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

    "topography": [False, True],
    "topographyzmax": [0.0, 1.0],
    "topographyorder": [2, 7],
    "topographyzetabreak": [None],
    "topographyinput": "gaussian",
    "topographyfile": [None],  # Unused
    "topographygaussianAmp": [0.1, 1.0],
    "topographygaussianXc": [0.1, 0.9],
    "topographygaussianYc": [0.1, 0.9],
    "topographygaussianLx": [0.1, 1.0],
    "topographygaussianLy": [0.1, 1.0],
    "topographyanalyticalMetric": [None],

    "rec": [False, True],
    "recx": [None],
    "recy": [None],
    "reclat": [-90.0, 90.0],
    "reclon": [-180.0, 180.0],
    "recz": [None],
    "recdepth": [None],
    "rectopodepth": [None],
    "recfile": [None],
    "recsta": [None],
    "recnsew": [0, 1],
    "recwriteEvery": [100, 10000],
    "recusgsformat": [0, 1],
    "recsacformat": [0, 1],
    "recvariables": ["displacement", "velocity", "div", "curl", "strains", "displacementgradient"],

    # No plans to test this.
    "checkpoint": [False],
    "checkpointtime": [None],
    "checkpointtimeInterval": [None],
    "checkpointcycle": [None],
    "checkpointcycleInterval": [None],
    "checkpointfile": [None],
    "checkpointbufsize": [None],

    # No plans to test this.
    "restart": [False],
    "restartfile": [None],
    "restartbufsize": [None],

    # No plans to test this.
    "dgalerkin": [False],
    "dgalerkinorder": [None],

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

    # No plans to test this.
    "testpointsource": [False],
    "testpointsourcecp": [None],
    "testpointsourcecs": [None],
    "testpointsourcerho": [None],
    "testpointsourcediractest": [None],
    "testpointsourcehalfspace": [None],
    "nodes": [1, 64],
    "tasks": [1, 32]}

default_params["nekbone"] = {"ifbrick": ".false.",
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
range_params["nekbone"] = {"ifbrick": [".false.", ".true."],
                           "iel0": [1, 1000],
                           "ielN": [1, 1000],
                           "istep": [1, 2, 4],
                           "nx0": [4, 12],
                           "nxN": [4, 12],
                           "nstep": [1, 2, 4],
                           "npx": [0, 1, 10],
                           "npy": [0, 1, 10],
                           "npz": [0, 1, 10],
                           "mx": [0, 1, 10],
                           "my": [0, 1, 10],
                           "mz": [0, 1, 10],
                           "nodes": [1, 64],
                           "tasks": [1, 32]}

# NOTE: Several of these parameters are commented out as they are unsupported in the MPI version of the program.
default_params["miniAMR"] = {"--help": False,
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
range_params["miniAMR"] = {"--help": [False],
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
                           "nodes": [1, 64],
                           "tasks": [1, 32]}

default_params["HACC-IO"] = {"numParticles": 20000,
                             "nodes": 1,
                             "tasks": 1}
range_params["HACC-IO"] = {"numParticles": [2000, 2000000],
                           "nodes": [1, 64],
                           "tasks": [1]}


# TODO: Remove/ignore this for deep learning.
def params_to_string(params):
    """ Convert the parameters list to a string.
    Used as comments on input files to make the parameters used clear.
    """
    string = ""
    for param in params:
        string += param + "=" \
            + str("None" if params[param] is None else params[param]) + ","
    return string


def get_next_index(app):
    """ Get the next unused test index of the associated app.
    Enables testing to resume while giving each test a unique index.
    """
    try:
        idx = int(max(os.listdir(TEST_DIR + app + "/"))) + 1
    except FileNotFoundError:
        idx = 0
    return idx


def make_slurm_script(f):
    """ Fill in the base contents of the SLURM script.
    Use format_map() to substitute parameters.
    """
    if "voltrino" in SYSTEM:
        contents = ('#!/bin/bash\n'
                    '#SBATCH -N {nodes}\n'
                    '#SBATCH --time=48:00:00\n'
                    '#SBATCH -J {app}\n'
                    '#SBATCH --exclusive --gres=craynetwork:0\n'
                    'export OMP_NUM_THREADS=1\n'
                    'export OMP_PLACES=threads\n'
                    'export OMP_PROC_BIND=true\n'
                    '/home/kmlamar/ProxyAppPredictor/ldms_init.sh $(pwd)\n'
                    'echo "----------------------------------------"\n'
                    'START_TS=$(date +"%s")\n'
                    'echo "----------------------------------------"\n'
                    # 'srun --ntasks-per-node={tasks} {command}\n'
                    'srun --exclusive --ntasks-per-node={tasks} {command}\n'
                    'echo "----------------------------------------"\n'
                    'END_TS=$(date +"%s")\n'
                    'DIFF=$(echo "$END_TS - $START_TS" | bc)\n'
                    'echo "timeTaken = $DIFF"\n'
                    'echo "----------------------------------------"').format_map(f)
    elif "eclipse" in SYSTEM or "ghost" in SYSTEM:
        contents = ('#!/bin/bash\n'
                    '#SBATCH -N {nodes}\n'
                    '#SBATCH --time=24:00:00\n'
                    #'#SBATCH --time=4:00:00\n'
                    '#SBATCH -J {app}\n'
                    #'#SBATCH -p batch\n'
                    #'#SBATCH -p short\n'
                    '#SBATCH --partition=batch,short\n'
                    '#SBATCH --account=fy140198\n'
                    '#SBATCH --exclusive\n'
                    'module load intel/20.0.2.254\n'
                    'module load openmpi-intel/4.0.3\n'
                    'module load gcc/4.9.3\n'
                    'module load fftw/intel-19\n'
                    'module load mkl/19.0.3.199\n'
                    'module load tools/papi-5.7.0\n'
                    'module load cmake/3.20.3\n'
                    'module load mkl/21.3\n'
                    'module load tools/papi-5.7.0\n'
                    'export LIBXML2_HOME=/projects/HPCMON/tools/libxml2_build\n'
                    'export BOOST_ROOT=/projects/HPCMON/tools/boost_1_71_0_build\n'
                    'export HDF5_ROOT=/projects/HPCMON/tools/hdf5-1.10.2_build/hdf5\n'
                    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/papi/install/cts1/papi-5.7.0/intel-18.0/lib\n'
                    'export OMP_PLACES=threads\n'  # NOTE: Only for ExaMiniMD?
                    'export OMP_PROC_BIND=true\n'  # NOTE: Only for ExaMiniMD?
                    '# run LDMS\n'
                    '/projects/ovis/UCF/ldms/ldms_init.sh $(pwd)\n'
                    'echo $SLURM_JOB_NODELIST\n'
                    'echo "----------------------------------------"\n'
                    'START_TS=$(date +"%s")\n'
                    'echo "----------------------------------------"\n'
                    'srun --exclusive --mpi=pmi2 --ntasks-per-node={tasks} {command}\n'
                    'echo "----------------------------------------"\n'
                    'END_TS=$(date +"%s")\n'
                    'DIFF=$(echo "$END_TS - $START_TS" | bc)\n'
                    'echo "timeTaken = $DIFF"\n'
                    'echo "----------------------------------------"').format_map(f)
    else:
        raise Exception("Invalid SYSTEM name provided: " + SYSTEM)
    return contents


def make_file(app, params):
    """ Create an input file for the relevant application.
    """
    contents = ""
    if app.startswith("ExaMiniMD"):
        contents += "# " + params_to_string(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        if params["lattice_constant"] is not None:
            contents += "lattice {lattice} {lattice_constant}\n".format_map(
                params)
        else:
            contents += "lattice {lattice} {lattice_offset_x} {lattice_offset_y} {lattice_offset_z}\n".format_map(
                params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(
            params)
        if params["ntypes"] is not None:
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
    elif app.startswith("LAMMPS"):
        contents += "# " + params_to_string(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        contents += "lattice {lattice} {lattice_constant}\n".format_map(params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(
            params)
        contents += "create_box {ntypes} box\n".format_map(params)
        contents += "create_atoms 1 box\n"
        contents += "mass {type} {mass}\n".format_map(params)
        contents += "velocity all create {temperature_target} {temperature_seed}\n".format_map(
            params)
        contents += "pair_style {force_type} {force_cutoff}\n".format_map(
            params)
        contents += "pair_coeff 1 1 1.0 1.0\n"
        contents += "neighbor {neighbor_skin} bin\n".format_map(params)
        contents += "neigh_modify delay 0 every {comm_exchange_rate} check no\n".format_map(
            params)
        contents += "fix 1 all nve\n"
        contents += "timestep {dt}\n".format_map(params)
        contents += "thermo {thermo_rate}\n".format_map(params)
        contents += "run {nsteps}\n".format_map(params)
    elif app == "sw4lite":
        contents += "# " + params_to_string(params) + "\n\n"
        sections = ["fileio", "grid", "time", "supergrid", "source", "block", "topography",
                    "rec", "checkpoint", "restart", "dgalerkin", "developer", "testpointsource"]
        for section in sections:
            if params[section]:
                contents += str(section) + " "
                for param in params:
                    if not param.startswith(section):
                        continue
                    if params[param] is None:
                        continue
                    if param == section:
                        continue
                    contents += str(param.partition(section)
                                    [2]) + "=" + str(params[param]) + " "
                contents += "\n"
            else:
                # If the section wasn't defined, be sure to wipe out all parameters associated with it.
                # This is important to keep training data pure.
                for param in params:
                    if param.startswith(section):
                        params[param] = None
    elif app == "nekbone":
        contents = ('{ifbrick} = ifbrick ! brick or linear geometry\n'
                    '{iel0} {ielN} {istep} = iel0,ielN(per proc),stride ! range of number of elements per proc.\n'
                    '{nx0} {nxN} {nstep} = nx0,nxN,stride ! poly. order range for nx1\n'
                    '{npx} {npy} {npz} = npx,npy,npz ! np distrb, if != np, nekbone handle\n'
                    '{mx} {my} {mz} = mx,my,mz ! nelt distrb, if != nelt, nekbone handle\n').format_map(params)
    return contents


def get_command(app, params):
    """ Build up the appropriate command for this application run.
    """
    # Get the executable.
    if "voltrino" in SYSTEM:
        exe = "/projects/ovis/UCF/voltrino_run/"
    elif "eclipse" in SYSTEM or "ghost" in SYSTEM:
            exe = "/projects/ovis/UCF/eclipse_run/"
    else:
        exe = "../../../"

    if app.startswith("ExaMiniMD"):
        exe += "ExaMiniMD" + "/" + "ExaMiniMD"
    elif app == "HACC-IO":
        exe += "HACC-IO" + "/" + "hacc_io"
    elif app.startswith("LAMMPS"):
        exe += "LAMMPS" + "/"
        if "voltrino" in SYSTEM:
            exe += "lmp_voltrino"
        elif "eclipse" in SYSTEM or "ghost" in SYSTEM:
            exe += "LAMMPS"
    elif "nekbone" in app:
        exe = "/projects/ovis/UCF/eclipse_apps/nekbone-3.1/test/example1/nekbone"
    else:
        exe += str(app) + "/" + str(app)
    # nekbone, LAMMPS and HACC-IO don't have debug builds.
    if DEBUG_APPS and app != "nekbone" and app != "LAMMPS" and app != "HACC-IO":
        exe += ".g"

    args = ""
    if app.startswith("ExaMiniMD"):
        args = "-il input.lj"
        args += " --comm-type MPI --kokkos-threads={tasks}".format_map(params)
    elif app.startswith("LAMMPS"):
        args = "-i in.ar.lj"
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
                if not params[param] or params[param] is None or params[param] == '':
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
            args += "--object {type} {bounce} {center_x} {center_y} {center_z} \
                {movement_x} {movement_y} {movement_z} {size_x} {size_y} \
                {size_z} {inc_x} {inc_y} {inc_z} ".format_map(params)
    elif app == "HACC-IO":
        args = "{numParticles} ".format_map(
            params)
        args += "/projects/ovis/UCF/data/" + SYSTEM + str(params["testNum"])

    # Assemble the whole command.
    command = exe + " " + args
    return command


def scrape_output(output, app, index):
    """ Scrape the output for runtime, errors, etc.
    """
    lines = output.split('\n')
    for line in lines:
        if "error" in line or "fatal" in line or "libhugetlbfs" in line:
            if "Plugin file not found" in line:
                continue
            if "Stale file handle" in line:
                continue
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
        if line.startswith("timeTaken = "):
            features[app][index]["timeTaken"] = \
                int(line[len("timeTaken = "):])
    return features


def append_test(app, test):
    """ Append the test results to a CSV file for later analysis.
    """
    # The location of the output CSV.
    output_file = TEST_DIR + app + "dataset.csv"
    # We only add the header if we are creating the file fresh.
    needs_header = not os.path.exists(output_file)
    # Make sure the error key is there. Otherwise, it'll be missing sometimes.
    # NOTE: We need to do this for any field that is not guaranteed to be
    # present in all tests. Additionally, things will break badly if new
    # features are added in the future. We must start from scratch in such a case.
    if "error" not in features[app][test].keys():
        features[app][test]["error"] = ""
    # Convert the test to a DataFrame.
    dataframe = pd.DataFrame(features[app][test], index=[
                             features[app][test]["testNum"]])
    # Append to CSV.
    dataframe.to_csv(output_file, mode='a', header=needs_header)


def generate_test(app, prod, index):
    """ Create a test instance and queue up a job to run it.
    """
    # These are the defaults right now.
    script_params = {"app": app,
                     "nodes": prod["nodes"],
                     "tasks": prod["tasks"]}

    # Get the default parameters, which we will adjust.
    params = copy.copy(default_params[app])
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
    test_path = Path(TEST_DIR + app + "/" + str(index).zfill(10))
    test_path.mkdir(parents=True, exist_ok=True)

    # NOTE: Add support for compiling per-test binaries from here, if needed.

    # Generate the input file contents.
    file_string = make_file(app, params)
    # If a file_string was generated
    if file_string != "":
        # Save the contents to an appropriately named file.
        if app.startswith("ExaMiniMD"):
            file_name = "input.lj"
        elif app.startswith("LAMMPS"):
            file_name = "in.ar.lj"
        elif app == "sw4lite":
            file_name = "input.in"
        elif app == "nekbone":
            file_name = "data.rea"
        with open(test_path / file_name, "w+", encoding="utf-8") as text_file:
            text_file.write(file_string)

    if app.startswith("ExaMiniMD") and params["force_type"] == "snap":
        # Copy in Ta06A.snap, Ta06A.snapcoeff, and Ta06A.snapparam.
        with open(test_path / "Ta06A.snap", "w+", encoding="utf-8") as text_file:
            text_file.write(SNAP_FILE)
        with open(test_path / "Ta06A.snapcoeff", "w+", encoding="utf-8") as text_file:
            text_file.write(SNAPCOEFF_FILE)
        with open(test_path / "Ta06A.snapparam", "w+", encoding="utf-8") as text_file:
            text_file.write(SNAPPARAM_FILE)

    # Get the full command, with executable and arguments.
    command = get_command(app, params)
    # Set the command in the parameters.
    # Everything else was set earlier.
    script_params["command"] = command

    if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
        # Generate the SLURM script contents.
        slurm_string = make_slurm_script(script_params)
        # Save the contents to an appropriately named file.
        with open(test_path / "submit.slurm", "w+", encoding="utf-8") as text_file:
            text_file.write(slurm_string)

    # TODO: Just before running the test, publish the parameters in JSON format.
    # json_params = copy.copy(params)
    # # TODO: Filter out parameters that don't help?
    # # json_params.pop("a bad param")
    # # Convert the dict to JSON.
    # json = json.dumps(json_params)
    # host = "localhost"
    # port = 412
    # # Run the command to publish the data.
    # ldmsd_stream_publish = "ldmsd_stream_publish -h {host} -x {xprt} -p {port} "\
    #                        "-a munge -s {name} -t {type} -f {fin} 2>&1" \
    #                        .format(host = host,
    #                                xprt = "sock",
    #                                port = port,
    #                                name = "app_params",
    #                                type = "json",
    #                                f_in = json)
    # output = subprocess.run(ldmsd_stream_publish, cwd=test_path, shell=True, check=False,
    #                         encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    # # DEBUG
    # print(str(output))

    # Queue up the job.
    queue_job(index, test_path, app, command)


def queue_job(index, test_path, app, command):
    """ Add a job to our Python queue.
    """
    # Ensure the queue doesn't get too big too fast.
    # Run some existing jobs until we are under the threshold.
    while len(queued_jobs) > MAX_QUEUE:
        run_job(lazy=False)

    queued_jobs[index] = {"test_path": test_path,
                          "app":      app,
                          "command":  command}
    run_job(lazy=True)
    return


def run_job(index=0, lazy=False):
    """ Run the job in SLURM/local.
    If lazy, only try queueing once.
    """
    global features

    if index != 0:
        job = queued_jobs[index]
    else:
        # If index = 0, pick a job at random from the queue.
        index, job = random.choice(list(queued_jobs.items()))

    # Wait until the test is ready to run.
    # On HPC, wait until the queue empties a bit.
    if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
        while True:
            # Get the number of jobs currently in my queue.
            n_jobs = int(subprocess.run("squeue -u kmlamar | grep -c kmlamar",
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        shell=True,
                                        check=False,
                                        encoding='utf-8').stdout)

            # If there is room on the queue, break out of the loop.
            # On my account, 5 jobs can run at once (MaxJobsPU),
            # Can check by running: sacctmgr show qos format=MaxJobsPU,MaxSubmitPU
            if n_jobs < MAX_JOBS:
                break

            if lazy:
                # If there is no room in the queue and we are lazy,
                # don't bother waiting and try again later.
                return False
            else:
                # DEBUG
                # print(str(len(queued_jobs)) + " jobs in queue.")

                # Wait before trying again.
                time.sleep(WAIT_TIME)
    # On local, do nothing.

    # Run the test case.
    # On HPC, submit the SLURM script.
    if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
        print("Queuing app: " + job["app"] + "\t test: " + str(index))
        output = subprocess.run("sbatch submit.slurm",
                                cwd=job["test_path"],
                                shell=True,
                                check=False,
                                encoding='utf-8',
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT).stdout
        # If the output doesn't match, something went wrong.
        if "Submitted batch job " not in output:
            print(output)
            return False
        job_id = int(output.split("Submitted batch job ", 1)[1])
        # Add the queued job to our wait list.
        # We add a dictionary so we can keep track of things when we
        # handle the output later.
        active_jobs[job_id] = {"app": job["app"],
                               "index": index, "path": job["test_path"]}
    # On local, run the command.
    else:
        print("Running app: " + job["app"] + "\t test: " + str(index))
        start = time.time()
        output = str(subprocess.run(job["command"],
                                    cwd=job["test_path"],
                                    shell=True,
                                    check=False,
                                    encoding='utf-8',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT).stdout)
        features[job["app"]][index]["timeTaken"] = time.time() - start
        # Save the command to file.
        with open(job["test_path"] / "command.txt", "w+", encoding="utf-8") as text_file:
            text_file.write(job["command"])
        # Save the output in the associated test's folder.
        with open(job["test_path"] / "output.txt", "w+", encoding="utf-8") as text_file:
            text_file.write(output)
        features = scrape_output(output, job["app"], index)

    # Remove the job from the list.
    queued_jobs.pop(index)

    return True


def finish_active_jobs(lazy=False):
    """ Handle any unfinished outputs.
    """
    global features

    if not lazy:
        # Ensure everything generated is in the active jobs list.
        while len(queued_jobs) > 0:
            run_job(lazy=False)

    # Keep running this loop until all active jobs have completed and been parsed.
    while len(active_jobs) > 0:
        # We want to finish a job each iteration.
        finished_a_job = False
        # Try to find a completed job in our active list.
        for job in active_jobs:
            # If the job is done, it will not be found in the queue.
            job_status = subprocess.run("squeue -j " + str(job),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        shell=True,
                                        check=False,
                                        encoding='utf-8').stdout
            # If the job is done.
            if "Invalid job id specified" in job_status \
                    or "kmlamar" not in job_status:
                # Open the file with the completed job.
                try:
                    f = open(active_jobs[job]["path"] /
                             ("slurm-" + str(job) + ".out"), "r", encoding="utf-8")
                    output = f.read()
                except IOError:
                    # The file likely doesn't exist yet.
                    # Try again later.
                    continue
                f.close()
                # Parse the output.
                features = scrape_output(
                    output, active_jobs[job]["app"], active_jobs[job]["index"])
                # Report an error to screen.
                if "error" in features[active_jobs[job]["app"]][active_jobs[job]["index"]]:
                    print(str(active_jobs[job]["app"]) + " " + str(active_jobs[job]["index"]) + ": " + str(
                        features[active_jobs[job]["app"]][active_jobs[job]["index"]]["error"]))
                else:
                    print(str(active_jobs[job]["app"]) + " " +
                          str(active_jobs[job]["index"]) + ": Completed!")
                # Save the output of this job to file.
                append_test(active_jobs[job]["app"], active_jobs[job]["index"])
                # The job has been parsed. Remove it from the list.
                active_jobs.pop(job)
                # We successfully finished a job.
                finished_a_job = True
                # We have found our job in the for loop.
                # Break out and start the search again.
                break
        if finished_a_job:
            # TODO: Queue another job.
            pass
        else:  # If we went through the whole queue and all remaining jobs were still active.
            # If we are lazily finishing jobs.
            if lazy:
                # Don't bother waiting. Break out now and come back later.
                break
            # Print the contents of the remaining queue.
            print(subprocess.run("squeue -u kmlamar",
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True,
                                 check=False,
                                 encoding='utf-8').stdout)
            # Wait before trying again.
            time.sleep(WAIT_TIME)


def adjust_params():
    """ This function tests the interactions between multiple parameters.
    It makes multiple adjustments to important parameters in the default file.
    The parameters support resuming from an existing set of tests.
    """
    # Loop through each Proxy App.
    for app in enabled_apps:
        # Some apps are very particular.
        # Don't bother with them and stick to randomly generated tests only.
        if app == "sw4lite" or app == "miniAMR":
            continue
        # Identify where we left off, in case we already have some results.
        resume_index = get_next_index(app)
        # Loop through each combination of parameter changes.
        # prod is the cartesian product of our adjusted parameters.
        for index, prod in enumerate((dict(zip(range_params[app], x)) for
                                      x in product(*range_params[app].values()))):
            # Skip iterations until we reach the target starting index.
            if resume_index > index:
                continue
            # Test skips and input hacks.
            if app.startswith("ExaMiniMD") or app.startswith("LAMMPS"):
                # A bit of a hack, but we don't really need to test all combinations of these.
                # Let lattice_nx dictate the values for lattice_ny and lattice_nz.
                prod["lattice_ny"] = prod["lattice_nx"]
                prod["lattice_nz"] = prod["lattice_nx"]
            elif app == "SWFFT":
                if SKIP_TESTS:
                    if prod["ngy"] is None and prod["ngz"] is not None:
                        # Skip this test. It is invalid.
                        print("Skipping invalid test " + str(index))
                        continue
            elif app == "nekbone":
                if SKIP_TESTS:
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
                if SKIP_TESTS:
                    skip = False
                    if prod["--uniform_refine"] == 1 and prod["--refine_freq"] != 0:
                        skip = True
                    if prod["--num_tsteps"] is not None and prod["--time"] is not None:
                        # skip = True
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

            generate_test(app, prod, index)
            # Try to finish jobs part-way.
            finish_active_jobs(lazy=True)
    finish_active_jobs(lazy=False)


def narrow_params():
    """ Run a narrow sweep of possible application parameters to root out any
    invalid parameter values.
    """
    # TODO: Sweep through this to identify heavy sw4lite parameters to avoid.
    # NOTE: Some of these parameters may do nothing in a narrow sweep like this.
    # Loop through each Proxy App.
    for app in enabled_apps:
        # For each parameter that can be changed, based on our range_params list.
        for param in range_params[app]:
            # Get the default parameters, which we will adjust slightly.
            params = copy.copy(default_params[app])
            # For each valid choice for this parameter.
            for choice in range_params[app][param]:
                # Skip this choice if it's already the default.
                if choice == default_params[app][param]:
                    continue
                # Replace the default with this choice.
                params[param] = choice
                # Run each test multiple times.
                for _ in range(REPEAT_COUNT):
                    # Get the index to save the test files.
                    index = get_next_index(app)
                    # Run the test.
                    generate_test(app, params, index)
                # Try to finish jobs part-way.
                finish_active_jobs(lazy=True)
    finish_active_jobs(lazy=False)


def read_df():
    """ Read an existing DataFrame back from a saved CSV.
    """
    # For each app.
    for app in enabled_apps:
        # Open the existing CSV.
        df[app] = pd.read_csv(TEST_DIR + app + "dataset.csv",
                              sep=",", header=0, index_col=0,
                              engine="c", quotechar="\"")
    return


def rand_param(app, param, values=''):
    """ Pick a random parameter value within a valid range.
    The approach used depends on the data type of the values."""
    if values == '':
        values = range_params[app][param]
    # If it is a boolean
    if isinstance(values[-1], bool):
        # Pick one of the values at random.
        return random.choice(values)
    # If it is a number:
    elif isinstance(values[-1], numbers.Number):
        # Get the lowest value.
        min_v = min(x for x in values if x is not None)
        # Get the highest value.
        max_v = max(x for x in values if x is not None)
        # Pick a random number between min and max to use as the parameter value.
        if isinstance(values[-1], float):
            return random.uniform(min_v, max_v)
        elif isinstance(values[-1], int):
            return random.randint(min_v, max_v)
        else:
            print("Found a range with type" + str(type(values[-1])))
            return random.randrange(min_v, max_v)
    # Else if it has no meaningful range (ex. str):
    else:
        # Pick one of the values at random.
        return random.choice(values)


def get_params(app):
    """ Get a random, valid test case for the given app.
    """
    params = {}

    # Confirm the number is a power of 2.
    def is_pow_2(x):
        return (x & (x-1) == 0) and x != 0
    # Round up to the nearest power of 2.

    def next_pow_2(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()
    
    def get_pow_2(upper,lower=1):
        """ Get a uniform random power of 2 between 1 and limit.
        """
        max_bits = upper.bit_length() - 1
        min_bits = lower.bit_length() - 1
        power = random.randint(min_bits, max_bits)
        return 2 ** power
    
    def factors(n):
            return set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

    # All jobs must set these.
    params["nodes"] = get_pow_2(max(x for x in range_params[app]["nodes"] if x is not None))
    params["tasks"] = get_pow_2(max(x for x in range_params[app]["tasks"] if x is not None))

    if app == "sw4lite":
        if random.choice(range_params[app]["fileio"]):
            params["fileio"] = True
            params["fileioverbose"] = rand_param(app, "fileioverbose")
            params["fileioprintcycle"] = rand_param(app, "fileioprintcycle")
        else:
            params["fileio"] = False

        params["grid"] = True

        def compute_end_grid_point(maxval, h):
            reltol = 1e-5
            abstol = 1e-12
            fnpts = round(maxval / h + 1)
            if math.fabs((fnpts - 1) * h - maxval) < reltol * math.fabs(maxval) + abstol:
                npts = int(fnpts)
            else:
                npts = int(fnpts) + 1
            return float(npts)
        if random.choice(range(2)):
            params["gridnx"] = rand_param(app, "gridnx")
            params["gridny"] = rand_param(app, "gridny")
            params["gridnz"] = rand_param(app, "gridnz")
            params["gridh"] = rand_param(app, "gridh")
            h = params["gridh"]
            xMax = (params["gridnx"]-1)*h
            yMax = (params["gridny"]-1)*h
            zMax = (params["gridnz"]-1)*h
        else:
            params["gridx"] = rand_param(app, "gridx")
            params["gridy"] = rand_param(app, "gridy")
            params["gridz"] = rand_param(app, "gridz")
            choice = random.choice(range(4))
            if choice == 0:
                params["gridh"] = rand_param(app, "gridh")
                h = params["gridh"]
                xMax = (compute_end_grid_point(params["gridx"], h)-1)*h
                yMax = (compute_end_grid_point(params["gridy"], h)-1)*h
                zMax = (compute_end_grid_point(params["gridz"], h)-1)*h
            elif choice == 1:
                params["gridnx"] = rand_param(app, "gridnx")
                h = params["gridx"]/(params["gridnx"]-1)
                xMax = (params["gridnx"]-1)*h
                yMax = (compute_end_grid_point(params["gridy"], h)-1)*h
                zMax = (compute_end_grid_point(params["gridz"], h)-1)*h
            elif choice == 2:
                params["gridny"] = rand_param(app, "gridny")
                h = params["gridy"]/(params["gridny"]-1)
                xMax = (compute_end_grid_point(params["gridx"], h)-1)*h
                yMax = (params["gridny"]-1)*h
                zMax = (compute_end_grid_point(params["gridz"], h)-1)*h
            elif choice == 3:
                params["gridnz"] = rand_param(app, "gridnz")
                h = params["gridz"]/(params["gridnz"]-1)
                xMax = (compute_end_grid_point(params["gridx"], h)-1)*h
                yMax = (compute_end_grid_point(params["gridy"], h)-1)*h
                zMax = (params["gridnz"]-1)*h

        params["time"] = True
        if random.choice(range(2)):
            params["timet"] = rand_param(app, "timet")
        else:
            params["timesteps"] = rand_param(app, "timesteps")

        if random.choice(range_params[app]["supergrid"]):
            params["supergrid"] = True
            params["supergridgp"] = rand_param(app, "supergridgp")
            params["supergriddc"] = rand_param(app, "supergriddc")
        else:
            params["supergrid"] = False

        # TODO: Support more than 1 call of this.
        params["source"] = True
        choice = random.choice(range(2))
        if random.choice(range(2)):
            params["sourcex"] = rand_param(app, "sourcex", [0.0, xMax])
            params["sourcey"] = rand_param(app, "sourcey", [0.0, yMax])
        else:
            params["sourcex"] = rand_param(app, "sourcex", [0.0, xMax])
            params["sourcey"] = rand_param(app, "sourcey", [0.0, yMax])
        if random.choice(range(2)):
            params["sourcez"] = rand_param(app, "sourcez", [0.0, zMax])
        else:
            params["sourcedepth"] = rand_param(app, "sourcedepth", [0.0, zMax])
        if random.choice(range(2)):
            if random.choice(range(2)):
                params["sourceFx"] = rand_param(app, "sourceFx")
            if random.choice(range(2)):
                params["sourceFy"] = rand_param(app, "sourceFy")
            if random.choice(range(2)):
                params["sourceFz"] = rand_param(app, "sourceFz")
            if "sourceFx" not in params and \
               "sourceFy" not in params and \
               "sourceFz" not in params:
                choice = random.choice(range(3))
                if choice == 0:
                    params["sourceFx"] = rand_param(app, "sourceFx")
                elif choice == 1:
                    params["sourceFy"] = rand_param(app, "sourceFy")
                elif choice == 2:
                    params["sourceFz"] = rand_param(app, "sourceFz")
            if random.choice(range(2)):
                params["sourcef0"] = rand_param(app, "sourcef0")
        else:
            if random.choice(range(2)):
                params["sourceMxx"] = rand_param(app, "sourceMxx")
            if random.choice(range(2)):
                params["sourceMxy"] = rand_param(app, "sourceMxy")
            if random.choice(range(2)):
                params["sourceMxz"] = rand_param(app, "sourceMxz")
            if random.choice(range(2)):
                params["sourceMyy"] = rand_param(app, "sourceMyy")
            if random.choice(range(2)):
                params["sourceMyz"] = rand_param(app, "sourceMyz")
            if random.choice(range(2)):
                params["sourceMzz"] = rand_param(app, "sourceMzz")
            if "sourceMxx" not in params and \
               "sourceMxy" not in params and \
               "sourceMxz" not in params and \
               "sourceMyy" not in params and \
               "sourceMyz" not in params and \
               "sourceMzz" not in params:
                choice = random.choice(range(6))
                if choice == 0:
                    params["sourceMxx"] = rand_param(app, "sourceMxx")
                elif choice == 1:
                    params["sourceMxy"] = rand_param(app, "sourceMxy")
                elif choice == 2:
                    params["sourceMxz"] = rand_param(app, "sourceMxz")
                elif choice == 3:
                    params["sourceMyy"] = rand_param(app, "sourceMyy")
                elif choice == 4:
                    params["sourceMyz"] = rand_param(app, "sourceMyz")
                elif choice == 5:
                    params["sourceMzz"] = rand_param(app, "sourceMzz")
            if random.choice(range(2)):
                params["sourcem0"] = rand_param(app, "sourcem0")
        params["sourcetype"] = rand_param(app, "sourcetype")
        params["sourcet0"] = rand_param(app, "sourcet0")
        if params["sourcetype"] != "Dirac":
            params["sourcefreq"] = rand_param(app, "sourcefreq")

        if random.choice(range_params[app]["topography"]):
            params["topography"] = True
            params["topographyinput"] = rand_param(app, "topographyinput")
            params["topographyzmax"] = rand_param(app, "topographyzmax")
            params["topographygaussianAmp"] = rand_param(
                app, "topographygaussianAmp")
            params["topographygaussianXc"] = rand_param(
                app, "topographygaussianXc")
            params["topographygaussianYc"] = rand_param(
                app, "topographygaussianYc")
            params["topographygaussianLx"] = rand_param(
                app, "topographygaussianLx")
            params["topographygaussianLy"] = rand_param(
                app, "topographygaussianLy")
            params["topographyzetabreak"] = rand_param(
                app, "topographyzetabreak")
            params["topographyorder"] = rand_param(app, "topographyorder")
            params["topographyfile"] = rand_param(app, "topographyfile")
            params["topographyanalyticalMetric"] = rand_param(
                app, "topographyanalyticalMetric")
        else:
            params["topography"] = False

        if random.choice(range_params[app]["block"]):
            params["block"] = True
            params["blockvp"] = rand_param(app, "blockvp")
            params["blockvs"] = rand_param(app, "blockvs")
            if random.choice(range(2)):
                params["blockrho"] = rand_param(app, "blockrho")
            else:
                params["blockr"] = rand_param(app, "blockr")
            if random.choice(range(2)):
                params["blockx1"] = rand_param(app, "blockx1", [0.0, xMax])
                params["blockx2"] = rand_param(app, "blockx2", [0.0, xMax])
                if params["blockx1"] > params["blockx2"]:
                    # Swap
                    tmp = params["blockx1"]
                    params["blockx1"] = params["blockx2"]
                    params["blockx2"] = tmp
            if random.choice(range(2)):
                params["blocky1"] = rand_param(app, "blocky1", [0.0, yMax])
                params["blocky2"] = rand_param(app, "blocky2", [0.0, yMax])
                if params["blocky1"] > params["blocky2"]:
                    # Swap
                    tmp = params["blocky1"]
                    params["blocky1"] = params["blocky2"]
                    params["blocky2"] = tmp
            if random.choice(range(2)):
                params["blockz1"] = rand_param(app, "blockz1", [0.0, zMax])
                params["blockz2"] = rand_param(app, "blockz2", [0.0, zMax])
                if params["blockz1"] > params["blockz2"]:
                    # Swap
                    tmp = params["blockz1"]
                    params["blockz1"] = params["blockz2"]
                    params["blockz2"] = tmp
            if "topography" in params:
                params["blockabsdepth"] = rand_param(app, "blockabsdepth")
            if random.choice(range(2)):
                params["blockrhograd"] = rand_param(app, "blockrhograd")
            if random.choice(range(2)):
                params["blockvpgrad"] = rand_param(app, "blockvpgrad")
            if random.choice(range(2)):
                params["blockvsgrad"] = rand_param(app, "blockvsgrad")
        else:
            params["block"] = False

        if random.choice(range_params[app]["rec"]):
            params["rec"] = True
            if random.choice(range(2)):
                params["recx"] = rand_param(app, "recx", [0.0, xMax])
                params["recy"] = rand_param(app, "recy", [0.0, yMax])
            else:
                params["reclat"] = rand_param(app, "reclat")
                params["reclon"] = rand_param(app, "reclon")
            choice = random.choice(range(3))
            if choice == 0:
                params["recz"] = rand_param(app, "recz", [0.0, zMax])
            elif choice == 1:
                params["recdepth"] = rand_param(app, "recdepth", [0.0, zMax])
            elif choice == 2:
                params["rectopodepth"] = rand_param(
                    app, "rectopodepth", [0.0, zMax])
            params["recfile"] = rand_param(app, "recfile")
            params["recsta"] = rand_param(app, "recsta")
            params["recnsew"] = rand_param(app, "recnsew")
            params["recwriteEvery"] = rand_param(app, "recwriteEvery")
            if random.choice(range(2)):
                params["recusgsformat"] = 1
                params["recsacformat"] = 0
            else:
                params["recusgsformat"] = 0
                params["recsacformat"] = 1
            params["recvariables"] = rand_param(app, "recvariables")
        else:
            params["rec"] = False

        # Ensure this is never enabled, as we have no plans to test it.
        params["developer"] = False

    elif app == "SWFFT":
        params["n_repetitions"] = rand_param(app, "n_repetitions")

        params["ngx"] = get_pow_2(max(x for x in range_params[app]["ngx"] if x is not None),
                                  min(x for x in range_params[app]["ngx"] if x is not None))

        if random.choice(range(2)):
            params["ngy"] = get_pow_2(max(x for x in range_params[app]["ngy"] if x is not None),
                                      min(x for x in range_params[app]["ngy"] if x is not None))
        else:
            params["ngy"] = None

        if params["ngy"] is not None and random.choice(range(2)):
            params["ngz"] = get_pow_2(max(x for x in range_params[app]["ngz"] if x is not None),
                                      min(x for x in range_params[app]["ngz"] if x is not None))
        else:
            params["ngz"] = None

    elif app == "miniAMR":
        params["--help"] = None

        params["--nx"] = rand_param(app, "--nx")
        if params["--nx"] % 2 != 0:
            params["--nx"] += 1
        params["--ny"] = rand_param(app, "--nx")
        if params["--ny"] % 2 != 0:
            params["--ny"] += 1
        params["--nz"] = rand_param(app, "--nx")
        if params["--nz"] % 2 != 0:
            params["--nz"] += 1

        params["--init_x"] = rand_param(app, "--init_x")
        params["--init_y"] = rand_param(app, "--init_x")
        params["--init_z"] = rand_param(app, "--init_x")

        params["--reorder"] = rand_param(app, "--reorder")

        processes_count = params["tasks"]
        proc_count = []
        proc_count.append(int(random.choice(list(factors(processes_count)))))
        proc_count.append(
            int(random.choice(list(factors(processes_count/proc_count[0])))))
        proc_count.append(int(processes_count/proc_count[0]/proc_count[1]))
        random.shuffle(proc_count)
        params["--npx"] = proc_count.pop()
        params["--npy"] = proc_count.pop()
        params["--npz"] = proc_count.pop()

        params["--max_blocks"] = rand_param(app, "--max_blocks")
        if params["--max_blocks"] < params["--init_x"] * \
                params["--init_y"] * params["--init_z"]:
            params["--max_blocks"] = params["--init_x"] * \
                params["--init_y"] * params["--init_z"]

        params["--num_refine"] = rand_param(app, "--num_refine")
        params["--block_change"] = rand_param(app, "--block_change")

        params["--uniform_refine"] = rand_param(app, "--uniform_refine")
        if params["--uniform_refine"] != 1:
            params["--refine_freq"] = rand_param(app, "--refine_freq")
        else:
            params["--refine_freq"] = None

        params["--inbalance"] = rand_param(app, "--inbalance")
        params["--lb_opt"] = rand_param(app, "--lb_opt")

        params["--num_vars"] = rand_param(app, "--num_vars")
        params["--comm_vars"] = rand_param(app,
                                           "--comm_vars", [0, params["--num_vars"]])

        if random.choice(range(2)):
            params["--num_tsteps"] = rand_param(app, "--num_tsteps")
        else:
            params["--time"] = rand_param(app, "--time")

        params["--stages_per_ts"] = rand_param(app, "--stages_per_ts")
        params["--permute"] = rand_param(app, "--permute")
        params["--code"] = rand_param(app, "--code")
        params["--checksum_freq"] = rand_param(app, "--checksum_freq")
        params["--stencil"] = random.choice(
            range_params["miniAMR"]["--stencil"])
        params["--error_tol"] = rand_param(app, "--error_tol")
        params["--report_diffusion"] = rand_param(app, "--report_diffusion")
        params["--report_perf"] = rand_param(app, "--report_perf")

        params["--num_objects"] = rand_param(app, "--num_objects")
        if params["--num_objects"] > 0:
            params["type"] = rand_param(app, "type")
            params["bounce"] = rand_param(app, "bounce")
            params["center_x"] = rand_param(app, "center_x")
            params["center_y"] = rand_param(app, "center_x")
            params["center_z"] = rand_param(app, "center_x")
            params["movement_x"] = rand_param(app, "movement_x")
            params["movement_y"] = rand_param(app, "movement_x")
            params["movement_z"] = rand_param(app, "movement_x")
            params["size_x"] = rand_param(app, "size_x")
            params["size_y"] = rand_param(app, "size_x")
            params["size_z"] = rand_param(app, "size_x")
            params["inc_x"] = rand_param(app, "inc_x")
            params["inc_y"] = rand_param(app, "inc_x")
            params["inc_z"] = rand_param(app, "inc_x")

    # TODO: Handle conditional cases for each app here.
    # elif app == "ExaMiniMDbase":
    #     pass
    # elif app == "ExaMiniMDsnap":
    #     pass
    # elif app == "LAMMPS":
    #     pass
    elif app == "nekbone":
        LELT = 1000
        params["ielN"] = rand_param(app, "ielN")
        assert(params["ielN"] <= LELT)
        params["iel0"] = rand_param(app, "iel0", [
            min(x for x in range_params[app]["iel0"] if x is not None),
            params["ielN"]])
        assert(params["iel0"] <= params["ielN"])
        params["istep"] = rand_param(app, "istep")
        LX1 = 12
        params["nxN"] = rand_param(app, "nxN")
        assert(params["nxN"] <= LX1)
        params["nx0"] = rand_param(app, "nx0", [
            min(x for x in range_params[app]["nx0"] if x is not None),
            params["nxN"]])
        assert(params["nx0"] <= params["nxN"])
        params["nstep"] = rand_param(app, "nstep")

        if random.choice(range(2)):
            # Automatically find the decomposition.
            params["npx"] = 0
            params["npy"] = 0
            params["npz"] = 0
            params["mx"] = 0
            params["my"] = 0
            params["mz"] = 0
        else:
            LP = 1000
            assert(params["nodes"] <= LP)
            processes_count = params["nodes"] # TODO: Verify this is the correct number.
            proc_count = []
            proc_count.append(int(random.choice(list(factors(processes_count)))))
            proc_count.append(
                int(random.choice(list(factors(processes_count/proc_count[0])))))
            proc_count.append(int(processes_count/proc_count[0]/proc_count[1]))
            random.shuffle(proc_count)
            params["npx"] = proc_count.pop()
            params["npy"] = proc_count.pop()
            params["npz"] = proc_count.pop()

            LELT = 1000
            assert(params["tasks"] <= LELT)
            processes_count = params["tasks"] # TODO: Verify this is the correct number.
            proc_count = []
            proc_count.append(int(random.choice(list(factors(processes_count)))))
            proc_count.append(
                int(random.choice(list(factors(processes_count/proc_count[0])))))
            proc_count.append(int(processes_count/proc_count[0]/proc_count[1]))
            random.shuffle(proc_count)
            params["mx"] = proc_count.pop()
            params["my"] = proc_count.pop()
            params["mz"] = proc_count.pop()

        # Set remaining parameters.
        for param, values in range_params[app].items():
            if param not in params:
                params[param] = rand_param(app, param)

    # The default case just picks parameters at random within a range.
    else:
        # For each parameter:
        for param, values in range_params[app].items():
            if param not in params:
                params[param] = rand_param(app, param)

    # Explicitly fill in unused parameters with None.
    # This is important to ensure default values aren't used,
    # to ensure CSV alignment, and to have some default value to train on.
    for param, values in range_params[app].items():
        if param not in params:
            params[param] = None

    return params


def random_tests():
    """ Run random permutations of tests.
    This extra variety helps training.
    """
    global terminate
    signal.signal(signal.SIGINT, exit_gracefully)
    # Cancel via Ctrl+C.
    # While we have not canceled the test:
    while not terminate:
        # Pick a random app.
        app = random.choice(list(enabled_apps))
        # Get the parameters.
        params = get_params(app)
        # Run each test multiple times.
        for i in range(REPEAT_COUNT):
            # Get the index to save the test files.
            index = get_next_index(app)
            # Run the test.
            generate_test(app, params, index)
        # Try to finish jobs.
        finish_active_jobs(lazy=True)
    # If we want to terminate, we can't be lazy. Be sure all jobs complete.
    finish_active_jobs(lazy=False)
    signal.signal(signal.SIGINT, original_sigint)


def pickle_data(data, name):
    if not os.path.exists(PICKLE_DIR):
        os.makedirs(PICKLE_DIR)
    pickle.dump(data, open(PICKLE_DIR + str(name), 'wb'))


def depickle_data(name):
    return pickle.load(open(PICKLE_DIR + str(name), 'rb'))


def regression(regressor, model_name, X, y, one_at_a_time=False):
    """ Train and test a regressor on a dataset.
    """
    ret = str(model_name) + "\n"

    # DEBUG
    # assert np.any(np.isinf(X)) and np.any(np.isnan(X)), "Invalid data in X"
    # assert np.any(np.isinf(y)) and np.any(np.isnan(y)), "Invalid data in y"

    # Plot the results for each regressor.
    y_pred = []
    # Some predictors only work properly when given one job at a time.
    if one_at_a_time:
        # Check for possible depickling.
        if should_depickle:
            y = depickle_data(model_name + "_y")
            y_pred = depickle_data(model_name + "_y_pred")
        else:
            assert len(X) == len(y)
            for i in range(len(X)):
                xVal = X.iloc[i]
                yVal = y.iloc[i]
                prediction = regressor.predict_and_fit(xVal, yVal)
                y_pred.append(prediction)

            # Pickle the data.
            pickle_data(y, model_name + "_y")
            pickle_data(y_pred, model_name + "_y_pred")

        # R^2 calculation since cross_val_score isn't meaningful here.
        numerator = ((y - y_pred) ** 2).sum(axis=0)
        denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0)
        score = 1 - numerator / denominator
        ret += " R^2: " + str(score) + "\n"
        plt.scatter(y_pred, y, s=20, c="black", label="data")
    else:
        # Check for possible depickling.
        if should_depickle:
            y = depickle_data(model_name + "_y")
            y_pred = depickle_data(model_name + "_y_pred")
            scores = depickle_data(model_name + "_scores")
            regressor = depickle_data(model_name + "_regressor")
            plot_regressor = depickle_data(model_name + "_plot__regressor")
            X = depickle_data(model_name + "_X")
            y = depickle_data(model_name + "_y")
            X_test = depickle_data(model_name + "_X_test")
            y_test = depickle_data(model_name + "_y_test")
            total_time = depickle_data(model_name + "_total_time")
        else:
            # Train Regressor.
            # Time the training.
            startTime = time.process_time()
            regressor = regressor.fit(X, y)
            endTime = time.process_time()
            total_time = endTime - startTime
            # Retrain on 4/5 of the data for plotting.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            plot_regressor = regressor.fit(X_train, y_train)

            scores = cross_val_score(regressor, X, y, cv=5,
                                 scoring="r2")

            y_pred = plot_regressor.predict(X_test)

            # Pickle the data.
            pickle_data(y, model_name + "_y")
            pickle_data(y_pred, model_name + "_y_pred")
            pickle_data(scores, model_name + "_scores")
            pickle_data(regressor, model_name + "_regressor")
            pickle_data(plot_regressor, model_name + "_plot__regressor")
            pickle_data(X, model_name + "_X")
            pickle_data(y, model_name + "_y")
            pickle_data(X_test, model_name + "_X_test")
            pickle_data(y_test, model_name + "_y_test")
            pickle_data(total_time, model_name + "_total_time")

        # Report cross-validation accuracy.
        ret += " R^2: " + str(scores.mean()) + "\n"
        ret += str(total_time) + "s \n"

        plt.scatter(y_pred, y_test, s=20, c="black", label="data")

    # plt.xlabel("Predicted (seconds) ("+str(model_name)+")")
    # plt.ylabel("Actual (seconds)")
    plt.xlabel("Predicted (minutes) ("+str(model_name)+")")
    plt.ylabel("Actual (minutes)")
    # if "ExaMiniMD" in model_name:
    #     plt.xscale('log',base=10)
    #     plt.yscale('log',base=10)
    # plt.legend()
    try:
        os.makedirs("figures")
    except FileExistsError:
        # Directory already exists
        pass
    plt.savefig("figures/"+str(model_name).replace(" ", "")+".svg")
    plt.close()

    if not one_at_a_time:
        if should_depickle:
            train_sizes = depickle_data(model_name + "_train_sizes")
            train_scores = depickle_data(model_name + "_train_scores")
            test_scores = depickle_data(model_name + "_test_scores")
        else:
            # Plot the learning curves.
            train_sizes, train_scores, test_scores = learning_curve(
                regressor,
                X,
                y,
                train_sizes=np.linspace(0.1, 1.0, 100),
                scoring="neg_root_mean_squared_error",
                cv=5,
                n_jobs=-1,
                # May be useful for training and testing run time measurements.
                # return_times=True,
            )
            pickle_data(train_sizes, model_name + "_train_sizes")
            pickle_data(train_scores, model_name + "_train_scores")
            pickle_data(test_scores, model_name + "_test_scores")
        
        plt.plot(train_sizes, -test_scores.mean(1), label=str(model_name))
        # Set the lower limit of the y-axis to 0
        plt.ylim(0, None)
        # Output learning curve data.
        if not os.path.exists("learning"):
            os.makedirs("learning")
        with open("learning/"+str(model_name)+".csv", "w+", encoding="utf-8") as learning_file:
            for i, train in enumerate(train_sizes):
                learning_file.write(
                    str(model_name)+","+str(train)+","+str(-test_scores.mean(1)[i])+"\n")
        plt.xlabel("Train size")
        plt.ylabel("Root Mean Squared Error")
        # plt.title("Learning curve - " + str(model_name))
        plt.legend(loc="best")
        plt.savefig("figures/"+str(model_name).replace(" ", "") +
                    "_learning.svg")
        plt.close()

    if "Decision Tree" in model_name:
        ret += str(model_name) + "Features:\n"
        ret += str(X.columns) + "\n" + \
            str(regressor.steps[1][1].feature_importances_)

    return ret


def run_regressor(X, y, preprocessor, model_idx, app="", only_count=False):
    """ Run a specific regressor for a specific application.
    When only_count=True, the number of regressors for the application is
    determined, but no regressors actually run.
    """

    def get_pipeline(preprocessor, clf):
        """ Convenience function to add a preprocessor to a regression pipeline.
        """
        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    # Make sure our features have the expected shape.
    # Also useful to keep track of test sizes.
    ret = str(X.shape) + "\n"
    # Columns must be maintained for some predictors.
    columns = list(X.columns)

    regressors = []

    # Run our regressors.
    regressors.append(
        (regression, get_pipeline(preprocessor, RandomForestRegressor()),
         "Random Forest Regressor "+app, X, y))

    regressors.append(
        (regression, get_pipeline(preprocessor, tree.DecisionTreeRegressor()),
         "Decision Tree Regressor "+app, X, y))

    regressors.append(
        (regression, get_pipeline(preprocessor, linear_model.BayesianRidge()),
         "Bayesian Ridge "+app, X, y))

    regressors.append(
        (regression, get_pipeline(preprocessor, svm.SVR()),
         "Support Vector Regression RBF "+app, X, y))
    for i in range(1, 3+1):
        regressors.append(
            (regression, get_pipeline(preprocessor, svm.SVR(kernel="poly", degree=i)),
             "Support Vector Regression poly "+str(i)+" "+app, X, y))
    regressors.append(
        (regression, get_pipeline(preprocessor, svm.SVR(kernel="sigmoid")),
         "Support Vector Regression sigmoid "+app, X, y))

    regressors.append(
        (regression, get_pipeline(preprocessor, make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))),
         "Linear Stochastic Gradient Descent Regressor "+app, X, y))

    for i in range(1, 7+1):
        regressors.append(
            (regression, get_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=i)),
             str(i)+" Nearest Neighbors Regressor "+app, X, y))

    if app != "nekbonebaseline":
        # NOTE: Anything larger than 2 is cut down to 2.
        for i in range(1, 2+1):
            regressors.append(
                (regression, get_pipeline(preprocessor, PLSRegression(n_components=i)),
                 str(i)+" PLS Regression "+app, X, y))

    for i in range(1, 2 + 1):
        layer_size = int((len(columns) + 1) / 2) # (Input size + output size) / 2
        layers = tuple(layer_size for _ in range(i))
        regressors.append(
            (regression, get_pipeline(preprocessor, MLPRegressor(activation="relu", hidden_layer_sizes=layers, random_state=1, max_iter=500)),
             str(i)+" MLP Regressor relu "+app, X, y))

    # NOTE: Preprocessing is bad for these. Use raw values.
    if app == "SWFFT":
        regressors.append(
            (regression, AnalyticalRegressor(columns), "Analytical Regressor "+app, X, y))
    regressors.append(
        (regression, TsafrirRegressor(columns), "Tsafrir Regressor "+app, X, y, True))
    regressors.append(
        (regression, ReqtimeRegressor(columns), "Reqtime Regressor "+app, X, y, True))
    regressors.append(
        (regression, CompleteRegressor(columns), "Complete Regressor "+app, X, y, True))

    if only_count:
        return len(regressors)

    print("Total model count: " + str(len(regressors)))
    if model_idx >= len(regressors):
        return ""
    # Run the selected regressor.
    ret += regressors[model_idx][0](*regressors[model_idx][1:])

    return ret


def ml(model_idx, app, baseline, only_count=False):
    """ Run machine learning on the DataFrames.
    When only_count=True, the number of regressors for the application is
    determined, but no regressors actually run.
    """
    global df

    def hasError(error):
        if not isinstance(error, str):
            return False
        lines = error.split('\n')
        for line in lines:
            if "Plugin file not found" in line:
                continue
            if "error" in line or "fatal" in line or "libhugetlbfs" in line:
                return True
        return False

    REMOVE_ERRORS = True

    # DEBUG
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Print the app name, so we keep track of which one is being worked on.
    print("\n" + app + ("baseline" if baseline else "") + "\n")
    X = df[app]

    # Use the error field to report simply whether or not we encountered an
    # error. We can use this as a training feature.
    if "error" in X.columns:
        if REMOVE_ERRORS:
            # Filter out errors.
            # Ignore cosmetic errors that don't hurt anything.
            X['error'] = X.apply(
                lambda row: hasError(row["error"]), axis=1)
            # Old approach that counts cosmetic errors.
            # X = X[X["error"].isnull()]
            X = X.drop(columns="error")
            if X.shape[0] < 1:
                print("All tests contained errors. Skipping...")
                return -1
        else:
            X["error"] = X["error"].notnull()

    # Simple replacements for base cases.
    X = X.replace('on', '1', regex=True)
    X = X.replace('off', '0', regex=True)
    X = X.replace('true', '1', regex=True)
    X = X.replace('false', '0', regex=True)
    X = X.replace('.true.', '1', regex=True)
    X = X.replace('.false.', '0', regex=True)

    # Choose what to predict.
    PREDICTION = "timeTaken"

    assert not (REMOVE_ERRORS and PREDICTION == "error")

    # Prediction selection.
    if PREDICTION == "error":
        # For predicting errors.
        y = X["error"].astype(float)
    elif PREDICTION == "timeTaken":
        # For predicting time taken
        y = X["timeTaken"].astype(float)
        # Prevent empty values for y.
        # This should never happen if tests complete gracefully.
        # Default to the max time of 48 hours.
        y = y.fillna(86400.0*2)

    # When predicting, time taken cannot be known ahead of time.
    X = X.drop(columns="timeTaken")
    if REMOVE_ERRORS:
        # The column was already removed by now in this case.
        pass
    else:
        # When predicting, we cannot know if the program crashed before it starts.
        X = X.drop(columns="error")
    # The testNum is also irrelevant for training purposes.
    X = X.drop(columns="testNum")

    if "ExaMiniMD" in app:
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
    if "LAMMPS" in app:
        for column in list(X.columns):
            if column not in range_params["LAMMPS"]:
                X = X.drop(columns=column)

    # Fill in inferred data, based on an understanding of how SWFFT works.
    if "SWFFT" in app:
        X["ngy"] = np.where(X["ngy"].isna(), X["ngx"], X["ngy"])
        X["ngz"] = np.where(X["ngz"].isna(), X["ngy"], X["ngz"])

    # Skip anything that isn't a job input parameter.
    if baseline:
        for col in X:
            if col not in ["nodes", "tasks"]:
                X = X.drop(columns=col)

    # X = scaler.transform(X)
    # # Feature selection. Removes useless columns to simplify the model.
    sel = feature_selection.VarianceThreshold(threshold=0)
    # X = sel.fit_transform(X)
    # # Discretization. Buckets results to whole minutes like related works.
    # # y = y.apply(lambda x: int(x/60))

    # Track the feature types of each cell.
    numeric_features = []
    categorical_features = []
    # Iterate over every cell.
    for col in X:
        # Identify what type each column is.
        isNumeric = True
        for rowIndex, row in X[col].iteritems():
            try:
                # If it can be a float, make it a float.
                X[col][rowIndex] = float(X[col][rowIndex])
                # If the float is NaN (unacceptable to Sci-kit), make it -1.0 for now.
                if pd.isnull(X[col][rowIndex]):
                    X[col][rowIndex] = -1.0
            except:
                # Otherwise, we will assume this is categorical data.
                isNumeric = False
        if isNumeric:
            # For whatever reason, float conversions don't want to work in Pandas dataframes.
            # Try changing the value column-wide instead.
            # TODO: Doesn't seem to actually solve anything.
            X[col] = X[col].astype(float)
            numeric_features.append(str(col))
        else:
            categorical_features.append(str(col))

    # Standardization for numeric data.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())])
    # One-hot encoding for categorical data.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    # Add the transformers to a preprocessor object.
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),])

    # Run regressors.
    result = run_regressor(X, y, preprocessor, model_idx,
                           app + ("baseline" if baseline else ""), only_count)
    print(result)
    return result


def base_test():
    """ Used to run a small set of hardcoded test cases.
    Useful for debugging purposes.
    """

    app = "sw4lite"
    params = copy.copy(default_params[app])
    generate_test(app, params, get_next_index(app))

    finish_active_jobs()
    return


def main():
    global range_params
    global should_depickle

    parser = argparse.ArgumentParser(
        description="Test and train a predictive framework.")
    parser.add_argument("--doML", action='store_true',
                        help="Whether or not to perform machine learning.")
    parser.add_argument("--fromCSV", action='store_true',
                        help="Whether or not to read from CSV")
    parser.add_argument("--modelIdx", type=int, required=False,
                        help="The index of the model")
    parser.add_argument("--app", type=str, choices=range_params.keys(), required=False,
                        help="App name")
    parser.add_argument("--baseline", type=int, choices=[0, 1], required=False,
                        help="Whether or not to run baseline (nodes and tasks only) tests")
    parser.add_argument("--depickle", action='store_true',
                        help="Whether to compute ML models or depickle already-made ones from file")
    args = parser.parse_args()
    doML = args.doML
    fromCSV = args.fromCSV
    model_idx = args.modelIdx
    app_name = args.app
    baseline = True if args.baseline == 1 else False
    should_depickle = args.depickle

    # Optionally start training from CSV immediately.
    if fromCSV:
        read_df()
    else:
        # Run through all of the primary tests.
        # adjust_params()
        # narrow_params()
        # Run tests at random indefinitely.
        random_tests()
        pass
    # Perform machine learning.
    if doML:
        if model_idx is not None and \
                app_name is not None:
            ml(model_idx, app_name, baseline)
        else:
            # TODO: make this concurrent.
            for base in baseline:
                for app in enabled_apps:
                    for model in range(ml(0, app, base, only_count=True)):
                        ml(model, app, base, only_count=False)


def exit_gracefully(signum, frame):
    """ Override the default signal handler to allow for graceful termination.
    ProxyAppPredictor will finish what's in its queue before closing.
    """
    global terminate
    # Restore the original signal handler as otherwise bad things will happen
    # in input when CTRL+C is pressed, and our signal handler is not re-entrant.
    signal.signal(signal.SIGINT, original_sigint)
    try:
        if input("\nReally quit? (y/n)> ").lower().startswith("y"):
            terminate = True
    except KeyboardInterrupt:
        print("Quitting...")
        terminate = True
    # Restore the exit gracefully handler here.
    signal.signal(signal.SIGINT, exit_gracefully)


if __name__ == "__main__":
    # store the original SIGINT handler.
    original_sigint = signal.getsignal(signal.SIGINT)
    # Run main.
    main()
