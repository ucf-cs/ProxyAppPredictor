"""A comprehensive script for generating, testing, and parsing inputs for a 
variety of Proxy Apps, both locally and on the Voltrino HPC testbed.
"""

import copy
import multiprocessing
import numbers
import numpy as np
import os
import pandas as pd
import platform
import random
import signal
import subprocess
import time

from itertools import product
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree
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
# TODO: Fix miniAMR.
enabledApps = rangeParams.keys()  # ["SWFFT", "nekbone", "miniAMR"]
# Whether or not to shortcut out tests that may be redundant or invalid.
skipTests = False
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
                                  "nsteps": 100}
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
                                  "nsteps": 100}
rangeParams["ExaMiniMD"] = {"force_type": ["lj/cut", "snap"],
                            "lattice_nx": [1, 5, 40, 100, 200, 500],
                            "dt": [0.0001, 0.0005, 0.001, 0.005, 1.0, 2.0],
                            "nsteps": [0, 10, 100, 1000]}

defaultParams["SWFFT"] = {"n_repetitions": 1,
                          "ngx": 512,
                          "ngy": None,
                          "ngz": None}
rangeParams["SWFFT"] = {"n_repetitions": [1, 2, 4, 8, 16, 64],
                        "ngx": [256, 512, 1024, 2048],
                        "ngy": [None, 256, 512, 1024, 2048],
                        "ngz": [None, 256, 512, 1024, 2048]}

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
                            "mz": 0}
rangeParams["nekbone"] = {"ifbrick": [".false.", ".true."],
                          "iel0": [1, 50],
                          "ielN": [50],
                          "istep": [1, 2],
                          "nx0": [1, 10],
                          "nxN": [10],
                          "nstep": [1, 2],
                          "npx": [0, 1, 10],  # TODO: Get np first.
                          "npy": [0, 1, 10],
                          "npz": [0, 1, 10],
                          "mx": [0, 1, 10],  # TODO: Get nelt first.
                          "my": [0, 1, 10],
                          "mz": [0, 1, 10]}

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
                            "inc_z": 0.0}
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
                          "--stages_per_ts": [0, 20],
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
                          "inc_z": [None]}


def paramsToString(params):
    string = ""
    for param in params:
        string += param + "=" \
            + str("None" if params[param] == None else params[param]) + ","
    return string


# Get the next unused index of the associated app.
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
    if app == "ExaMiniMD":
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
    if app == "ExaMiniMD":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/ExaMiniMD/ExaMiniMD"
        else:
            exe = "../../../ExaMiniMD"
    elif app == "SWFFT":
        if SYSTEM == "voltrino-int":
            exe = "/projects/ovis/UCF/voltrino_run/SWFFT/SWFFT"
        else:
            exe = "../../../SWFFT"
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
    if app == "ExaMiniMD":
        if SYSTEM == "voltrino-int":
            thread = 1
        else:
            thread = multiprocessing.cpu_count()
        args = "-il input.lj --comm-type MPI --kokkos-threads={}".format(
            thread)
    elif app == "SWFFT":
        # Locally adjust the params list to properly handle None.
        for param in params:
            params[param] = params[param] is not None and params[param] or ''
        args = "{n_repetitions} {ngx} {ngy} {ngz}".format_map(params)
    elif app == "nekbone":
        args = ""
    elif app == "miniAMR":
        for param in params:
            # Each of our standard parameters starts with "--".
            if param.startswith("--"):
                # If the parameter is unset, don't add it to the args list.
                if params[param] is False or None or '':
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
            args += "--object {type} {bounce} {center_x} {center_y} {center_z} {movement_x} {movement_y} {movement_z} {size_x} {size_y} {size_z} {inc_x} {inc_y} {inc_z} ".format_map(
                params)

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
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
        if "libhugetlbfs" in line:
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
    return features


def generateTest(app, prod, index):
    global activeJobs
    global features

    # Add any test skips and input hacks here.
    if app == "ExaMiniMD":
        # A bit of a hack, but we don't really need to test all combinations of these.
        # Let lattice_nx dictate the values for lattice_ny and lattice_nz.
        prod["lattice_ny"] = prod["lattice_nx"]
        prod["lattice_nz"] = prod["lattice_nx"]
    elif app == "SWFFT":
        if skipTests:
            if prod["ngy"] == None and prod["ngz"] != None:
                # Skip this test. It is invalid.
                print("Skipping invalid test " + str(index))
                return False
    elif app == "nekbone":
        if skipTests:
            skip = False
            if prod["iel0"] > prod["ielN"]:
                skip = True
            if prod["nx0"] > prod["nxN"]:
                skip = True
            if skip:
                print("Skipping invalid test " + str(index))
                return False
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
                skip = True
            if prod["load"] == "hilbert" and prod["--reorder"] == 1:
                skip = True
            if prod["--max_blocks"] < (prod["--init_x"] * prod["--init_y"] * prod["--init_z"]):
                skip = True
            if prod["load"] != "rcb" and prod["--change_dir"] == True:
                skip = True
            if prod["load"] != "rcb" and prod["--break_ties"] == True:
                skip = True

            if skip:
                print("Skipping redundant test " + str(index))
                return False

    # Get the default parameters, which we will adjust.
    # ExaMiniMD uses multiple sets of sane defaults based on force type.
    if app == "ExaMiniMD":
        if prod["force_type"] == "lj/cut":
            params = copy.copy(defaultParams[app + "base"])
        elif prod["force_type"] == "snap":
            params = copy.copy(defaultParams[app + "snap"])
        else:
            params = copy.copy(defaultParams[app])
    else:
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
        if app == "ExaMiniMD":
            fileName = "input.lj"
        elif app == "nekbone":
            fileName = "data.rea"
        with open(testPath / fileName, "w+") as text_file:
            text_file.write(fileString)

    if app == "ExaMiniMD" and params["force_type"] == "snap":
        # Copy in Ta06A.snap, Ta06A.snapcoeff, and Ta06A.snapparam.
        with open(testPath / "Ta06A.snap", "w+") as text_file:
            text_file.write(snapFile)
        with open(testPath / "Ta06A.snapcoeff", "w+") as text_file:
            text_file.write(snapcoeffFile)
        with open(testPath / "Ta06A.snapparam", "w+") as text_file:
            text_file.write(snapparamFile)

    # Get the full command, with executable and arguments.
    command = getCommand(app, params)

    if SYSTEM == "voltrino-int":
        # These are the defaults right now.
        # TODO: Consider adjusting these and treating them as a separate set of parameters.
        scriptParams = {"app": app,
                        "command": command,
                        "nodes": 4,
                        "tasks": 32}
        # nekbone is limited to 10 processors. It gets different rules.
        if app == "nekbone":
            scriptParams["nodes"] = 1
            scriptParams["tasks"] = 10
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
                print("Parsing output from job: " + str(job)
                      + "\tapp: " + activeJobs[job]["app"]
                      + "\ttest: " + str(activeJobs[job]["index"]))
                # Open the file with the completed job.
                with open(activeJobs[job]["path"] / ("slurm-" + str(job) + ".out"), "r") as f:
                    output = f.read()
                # Parse the output.
                features = scrapeOutput(
                    output, activeJobs[job]["app"], activeJobs[job]["index"])
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
def adjustParams(startApp="", startIdx=0):
    global features
    global df

    # Loop through each Proxy App.
    for app in enabledApps:
        # If a start app was specified.
        if len(startApp) != 0:
            # And the start app has not yet been reached.
            if app != startApp:
                continue
            # One we reach the starting app.
            else:
                # Remove the requirement so subsequent apps will be checked.
                startApp = ""
        # Loop through each combination of parameter changes.
        # prod is the cartesian product of our adjusted parameters.
        for index, prod in enumerate((dict(zip(rangeParams[app], x)) for
                                      x in product(*rangeParams[app].values()))):
            # Skip iterations until we reach the target starting index.
            if startIdx > index:
                continue
            generateTest(app, prod, index)
        if startIdx != 0:
            # Ensure subsequent loops start at the beginning.
            startIdx = 0

    finishJobs()

    # Convert each app dictionary to a DataFrame.
    # TODO: Append to the CSV as jobs complete. This way, we don't have to worry about losing everything if the script terminates early. This will also allow us to keep adding to our test dataset as we go.
    # dataframe.to_csv("sales.csv", index=False, mode='a', header=False)
    for app in enabledApps:
        print("Saving DataFrame for app: " + app)
        df[app] = pd.DataFrame(features[app])
        # Save parameters and results to CSV for optional recovery.
        df[app].to_csv("./tests/" + app + "/dataset.csv")


# Read an existing DataFrame back from a saved CSV.
def readDF():
    global df

    # For each app.
    for app in enabledApps:
        # Open the existing CSV.
        df[app] = pd.read_csv("./tests/" + app + "/dataset.csv", index_col=0)
    return


# Run random permutations of tests outside of the specific set of tests we have defined. This extra variety helps training.
def randomTests():
    global terminate
    # Cancel via Ctrl+C.
    # While we have not canceled the test:
    while not terminate:
        # Pick a random app.
        app = random.choice(list(rangeParams))
        # Get the index to save the test files.
        index = getNextIndex(app)
        # A parameters list to populate.
        params = {}
        # For each parameter:
        for param, values in rangeParams[app].items():
            # If it is a number:
            if isinstance(values[-1], numbers.Number):
                # Get lowest value.
                minV = min(x for x in values if x is not None)
                # Get highest value.
                maxV = max(x for x in values if x is not None)
                # Pick a random number between min and max to use as the parameter value.
                if isinstance(values[-1], float):
                    params[param] = random.uniform(minV, maxV)
                elif isinstance(values[-1], int):
                    params[param] = random.randint(minV, maxV)
                else:
                    print("Found a range with type" + str(type(values[-1])))
                    params[param] = random.randrange(minV, maxV)
            # Else if it has no meaningful range (ex. str):
            else:
                # Pick one of the values at random.
                params[param] = random.choice(values)
        # Run the test.
        generateTest(app, params, index)


# Train and test a regressor on a dataset.
def regression(regressor, modelName, X, y):
    print(modelName)
    # Train Regressor.
    regressor = regressor.fit(X, y)
    # Run and report cross-validation accuracy.
    scores = cross_val_score(regressor, X, y, cv=5)
    print("Accuracy: " + str(scores.mean()))
    return


# Run machine learning on the DataFrames.
def ML():
    global df

    for app in enabledApps:
        # Print the app name, so we keep track of which one is being worked on.
        print("\n" + app)
        # Transpose the data.
        X = df[app].T

        # Special cases to fill in missing data.
        if app == "SWFFT":
            X = X[X["ngy"].notnull()]
            X = X[X["ngz"].notnull()]
        #     print(X)
        #     X["ngy"] = X["ngy"].fillna(X["ngx"])
        #     X["ngz"] = X["ngz"].fillna(X["ngx"])
        #     print(X)

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
        le = LabelEncoder()
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
        # The time taken should be an output, not an input.
        y = X["timeTaken"].astype(float)
        X = X.drop(columns="timeTaken")
        # The testNum is also irrelevant for training purposes.
        X = X.drop(columns="testNum")

        # Make sure our features have the expected shape.
        # Also useful to keep track of test sizes.
        print(X.shape)

        # Run our regressors.
        # For now, these are a bunch of off-the-shelf regressors with default settings.
        forestRegressor = RandomForestRegressor()
        regression(forestRegressor, "Random Forest Regressor", X, y)
        regression(linear_model.BayesianRidge(), "Bayesian Ridge", X, y)
        regression(svm.SVR(), "Support Vector Regression", X, y)
        regression(make_pipeline(StandardScaler(), SGDRegressor(
            max_iter=1000, tol=1e-3)), "Linear Stochastic Gradient Descent Regressor", X, y)
        regression(KNeighborsRegressor(n_neighbors=2),
                   "K Nearest Neighbors Regressor", X, y)
        regression(PLSRegression(n_components=2), "PLS Regression", X, y)
        regression(tree.DecisionTreeRegressor(),
                   "Decision Tree Regressor", X, y)
        regression(MLPRegressor(random_state=1, max_iter=500),
                   "MLP Regressor", X, y)

        # Plot the results for the random forest regressor.
        y_1 = forestRegressor.predict(X)
        plt.figure()
        plt.scatter(y_1, y, s=20, edgecolor="black",
                    c="darkorange", label="data")
        plt.xlabel("Predicted (RF)")
        plt.ylabel("Actual")
        plt.legend()
        plt.show()


def main():
    fromCSV = False

    # Optionally start training from CSV immediately.
    if fromCSV:
        readDF()
    else:
        # Run through all of the primary tests.
        #adjustParams("", 0)
        # Run tests at random indefinitely.
        randomTests()
    # Perform machine learning.
    ML()


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
    signal.signal(signal.SIGINT, exit_gracefully)
    # Run main.
    main()
