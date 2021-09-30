"""A comprehensive script for generating, testing, and parsing inputs for a 
variety of Proxy Apps, both locally and on the Voltrino HPC testbed.
"""

import copy
import multiprocessing
import pandas as pd
import platform
import re
import subprocess
import time
from itertools import product
from pathlib import Path

# The default parameters for each application.
defaultParams = {}
# A set of sane parameter ranges.
# Our intent is to sweep through these with different input files.
# TODO: May want to consider implicit cases where a parameter is missing and has some default value?
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
enabledApps = rangeParams.keys()#["miniAMR"]
# Whether or not to shortcut out tests that may be redundant or invalid.
skipTests = False

# A set of sane defaults based on 3d Lennard-Jones melt (in.lj).
defaultParams["ExaMiniMD"] = {"units": "lj",
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
                              "comm_newton": "on",
                              "nsteps": 100}
rangeParams["ExaMiniMD"] = {"lattice_nx": [1, 5, 40, 100, 200],
                            "dt": [0.0001, 0.001, 0.005, 1.0, 2.0],
                            "nsteps": [0, 10, 100, 1000]}

defaultParams["SWFFT"] = {"n_repetitions": 1,
                          "ngx": 512,
                          "ngy": None,
                          "ngz": None}
rangeParams["SWFFT"] = {"n_repetitions": [1, 2, 4, 8, 16],
                        "ngx": [256, 512, 1024, 2048],
                        "ngy": [None, 256, 512, 1024, 2048],
                        "ngz": [None, 256, 512, 1024, 2048]}

defaultParams["miniAMR"] = {"--help": False,
                            "--nx": 10,
                            "--ny": 10,
                            "--nz": 10,
                            "--init_x": 1,
                            "--init_y": 1,
                            "--init_z": 1,
                            "--reorder": 1, # NOTE: Default depends on load parameter. 1 is default for RCB.
                            "load": "rcb",
                            "--npx": 1,
                            "--npy": 1,
                            "--npz": 1,
                            "--max_blocks": 500,
                            "--num_refine": 5,
                            "--block_change": 5, # = num_refine.
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
                            "--blocking_send": False,
                            "--code": 0,
                            "--checksum_freq": 5,
                            "--stencil": 7,
                            "--error_tol": 8,
                            "--report_diffusion": False,
                            "--report_perf": 1,
                            "--refine_ghosts": False,
                            "--send_faces": False,
                            "--change_dir": False,
                            "--group_blocks": False,
                            "--break_ties": False,
                            "--limit_move": False,
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
                            "load": ["rcb", "morton", "hilbert", "trunc_hilbert"],
                            "--npx": [1, 3],
                            "--npy": [None],
                            "--npz": [None],
                            "--max_blocks": [5, 500],
                            "--num_refine": [0, 5],
                            "--block_change": [0, 5],
                            "--uniform_refine": [0, 1],
                            "--refine_freq": [1, 5], # Ignored if uniform_refine=1
                            "--inbalance": [0, 50],
                            "--lb_opt": [0, 1, 2],
                            "--num_vars": [1, 40],
                            "--comm_vars": [0, 40],
                            "--num_tsteps": [None, 20],
                            "--time": [None, 20], # Ignored if num_tsteps is used.
                            "--stages_per_ts": [0, 20],
                            "--permute": [True, False],
                            "--blocking_send": [True, False],
                            "--code": [0, 1, 2],
                            "--checksum_freq": [0, 5],
                            "--stencil": [0, 7, 27],
                            "--error_tol": [0, 8],
                            "--report_diffusion": [True, False],
                            "--report_perf": [0],#list(range(0,15+1)), # Can be set to 0 in tests to disable output.
                            "--refine_ghosts": [True, False],
                            "--send_faces": [True, False],
                            "--change_dir": [True, False],
                            "--group_blocks": [True, False],
                            "--break_ties": [True, False],
                            "--limit_move": [True, False],
                            "--num_objects": [0,5],#list(range(0,5)),
                            "type": list(range(0,25+1)),
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
            + str("None" if params[param]==None else params[param]) + ","
    return string


def makeSLURMScript(f):
    # The base contents of the SLURM script.
    # Use format_map() to substitute parameters.
    contents = ('#!/bin/bash\n'
                '#SBATCH -N 4\n'
                '#SBATCH --time=24:00:00\n'
                '#SBATCH -J {app}\n'
                'export OMP_NUM_THREADS=1\n'
                'export OMP_PLACES=threads\n'
                'export OMP_PROC_BIND=true\n'
                'echo "----------------------------------------"\n'
                'START_TS=$(date +"%s")\n'
                'echo "----------------------------------------"\n'
                'srun --ntasks-per-node=32 {command}\n'
                'echo "----------------------------------------"\n'
                'END_TS=$(date +"%s")\n'
                'DIFF=$(echo "$END_TS - $START_TS" | bc)\n'
                'echo "timeTaken = $DIFF"\n'
                'echo "----------------------------------------"').format_map(f)
    return contents


def makeFile(app, params):
    contents = ""
    if app=="ExaMiniMD":
        contents += "# " + paramsToString(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        if params["lattice_constant"] != None:
            contents += "lattice {lattice} {lattice_constant}\n".format_map(params)
        else:
            contents += "lattice {lattice} {lattice_offset_x} {lattice_offset_y} {lattice_offset_z}\n".format_map(params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(params)
        if params["ntypes"] != None:
            contents += "create_box {ntypes}\n".format_map(params)
        contents += "create_atoms\n"
        contents += "mass {type} {mass}\n".format_map(params)
        if params["force_type"] != "snap":
            contents += "pair_style {force_type} {force_cutoff}\n".format_map(params)
        else:
            contents += "pair_style {force_type}\n".format_map(params)
        contents += "velocity all create {temperature_target} {temperature_seed}\n".format_map(params)
        contents += "neighbor {neighbor_skin}\n".format_map(params)
        contents += "neigh_modify every {comm_exchange_rate}\n".format_map(params)
        contents += "fix 1 all nve\n"
        contents += "thermo {thermo_rate}\n".format_map(params)
        contents += "timestep {dt}\n".format_map(params)
        contents += "newton {comm_newton}\n".format_map(params)
        contents += "run {nsteps}\n".format_map(params)
    return contents


def getCommand(app, params):
    # Get the executable.
    # NOTE: Is there a better way than hardcoding these into the function?
    # Does it really matter anyway? I think not.
    if app == "ExaMiniMD":
        if SYSTEM == "voltrino":
            exec = "/projects/ovis/UCF/voltrino_run/ExaMiniMD/ExaMiniMD"
        else:
            exec = "../../../ExaMiniMD"
    elif app == "SWFFT":
        if SYSTEM == "voltrino":
            exec = "/projects/ovis/UCF/voltrino_run/SWFFT/SWFFT"
        else:
            exec = "../../../SWFFT"
    elif app == "miniAMR":
        if SYSTEM == "voltrino":
            exec = "/projects/ovis/UCF/voltrino_run/miniAMR/miniAMR.x"
        else:
            exec = "../../../miniAMR.x"

    args = ""
    if app == "ExaMiniMD":
        if SYSTEM == "voltrino":
            thread = 1
        else:
            thread = multiprocessing.cpu_count()
        args = "-il input.lj --comm-type MPI --kokkos-threads={}".format(thread)
    elif app == "SWFFT":
        # Locally adjust the params list to properly handle None.
        for param in params:
            params[param] = params[param] is not None and params[param] or ''
        args = "{n_repetitions} {ngx} {ngy} {ngz}".format_map(params)
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
        # The object parameter is required.
        # Fill in each of these arguments.
        args += "--object {type} {bounce} {center_x} {center_y} {center_z} {movement_x} {movement_y} {movement_z} {size_x} {size_y} {size_z} {inc_x} {inc_y} {inc_z}".format_map(params)


    # Assemble the whole command.
    command = exec + " " + args
    return command


# Scrape the output for runtime, errors, etc.
def scrapeOutput(features, output, app, index):
    lines = output.split('\n')
    for line in lines:
        if line.startswith("timeTaken = "):
            features[app][index]["timeTaken"] = \
                int(line[len("timeTaken = "):])
        if line.startswith("error:"):
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
        if line.startswith("libhugetlbfs"):
            if "error" not in features[app][index].keys():
                features[app][index]["error"] = ""
            features[app][index]["error"] += line + "\n"
    return features


def generateTest(app, prod, index):
    global activeJobs
    global features

    # Add any test skips, input hacks, here.
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
    params = copy.copy(defaultParams[app])
    # Update the params based on our cartesian product.
    params.update(prod)
    # Add the test number to the list of params.
    params["testNum"] = index

    # Add the parameters to a DataFrame.
    features[app][index] = params

    # Create a folder to hold a SLURM script and any input files needed.
    testPath = Path("./tests/" + app + "/" + str(index).zfill(10))
    testPath.mkdir(parents=True,exist_ok=True)

    # Generate the input file contents.
    # TODO: Handle cases where apps require the generation of multiple input files.
    fileString = makeFile(app, params)
    # If a fileString was generated
    if fileString != "":
        # Save the contents to an appropriately named file.
        with open(testPath / "input.lj", "w+") as text_file:
            text_file.write(fileString)

    # Get the full command, with executable and arguments.
    command = getCommand(app, params)

    if SYSTEM == "voltrino":
        # Generate the SLURM script contents.
        SLURMString = makeSLURMScript({"app": app, "command": command})
        # Save the contents to an appropriately named file.
        with open(testPath / "submit.slurm", "w+") as text_file:
            text_file.write(SLURMString)

    # Wait until the test is ready to run.
    # On Voltrino, wait until the queue empties a bit.
    if SYSTEM == "voltrino":
        while True:
            # Get the number of jobs currently in my queue.
            nJobs = int(subprocess.run("squeue -u kmlamar | grep -c kmlamar", \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, \
                shell=True, encoding='utf-8').stdout)
            # If there is room on the queue, break out of the loop.
            # On my account, 5 jobs can run at once (MaxJobsPU),
            # 10 can be queued (MaxSubmitPU).
            if nJobs < 10:
                break
            # Wait before trying again.
            time.sleep(WAIT_TIME)
    # On local, do nothing.
    
    # Run the test case.
    # On Voltrino, submit the SLURM script.
    if SYSTEM == "voltrino":
        print("Queuing app: " + app + "\t test: " + str(index))
        output = subprocess.run("sbatch submit.slurm", cwd=testPath, \
            shell=True, encoding='utf-8', stdout=subprocess.PIPE, \
            stderr=subprocess.STDOUT).stdout
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
        output = str(subprocess.run(command, cwd=testPath, shell=True, \
            encoding='utf-8', stdout=subprocess.PIPE, \
            stderr=subprocess.STDOUT).stdout)
        features[app][index]["timeTaken"] = time.time() - start
        # Save the command to file.
        with open(testPath / "command.txt", "w+") as text_file:
            text_file.write(command)
        # Save the output in the associated test's folder.
        with open(testPath / "output.txt", "w+") as text_file:
            text_file.write(output)
        features = scrapeOutput(features, output, app, index)
    return True


# Handle any unfinished outputs.
def finishJobs():
    global activeJobs
    global features

    while len(activeJobs) > 0:
        for job in activeJobs:
            # If the job is done, it will not be found in the queue.
            if subprocess.run("squeue -j " + str(job), \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, \
                encoding='utf-8').stdout == \
                "slurm_load_jobs error: Invalid job id specified":
                
                print("Parsing output from job: " + job \
                    + "\tapp: " + activeJobs[job]["app"] \
                    + "\ttest: " + activeJobs[job]["index"])
                # Open the file with the completed job.
                with open() as f:
                    output = f.read(activeJobs[job]["path"] / "slurm-" + job \
                        + ".out")
                features = scrapeOutput(features, output, \
                    activeJobs[job]["app"], activeJobs[job]["index"])
                # The job has been parsed. Remove it from the list.
                activeJobs.pop(job)
                # Must break now that the dictionary size changed mid-iteration.
                break
        # Print the current queue status.
        print(subprocess.run("squeue -u kmlamar",\
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, \
            encoding='utf-8').stdout)
        # Wait before trying again.
        time.sleep(WAIT_TIME)


def slightAdjustParams():
    global features

    # Loop through each Proxy App.
    for app in enabledApps:
        # Initialize the app's dictionary.
        features[app] = {}
        index = 0
        # For each parameter that can be changed, based on our rangeParams list.
        for param in rangeParams[app]:
            # Get the default parameters, which we will adjust slightly.
            params = copy.copy(defaultParams[app])
            # For each valid choice for this parameter.
            for choice in rangeParams[app][param]:
                # Skip this choice if it's already the default.
                if choice == defaultParams[app][param]:
                    continue
                # Replace the default with this choice.
                params[param] = choice
                # Run the test.
                generateTest(app, params, index)
                index += 1
    
    finishJobs()

    # Convert each app dictionary to a DataFrame.
    for app in enabledApps:
        print("Saving DataFrame for app: " + app)
        df[app] = pd.DataFrame(features[app])
        # Save parameters and results to CSV for optional recovery.
        df[app].to_csv("./tests/" + app + "/dataset.csv")


# This function tests the interactions between multiple parameters.
# It makes multiple adjustments to important parameters in the default file.
def wideAdjustParams():
    global features

    # Loop through each Proxy App.
    for app in enabledApps:
        # Initialize the app's dictionary.
        features[app] = {}
        # Loop through each combination of parameter changes.
        # prod is the cartesian product of our adjusted parameters.
        # TODO: Perform multiple runs for each input? Perhaps 5?
        for index, prod in enumerate((dict(zip(rangeParams[app], x)) for \
            x in product(*rangeParams[app].values()))):
            generateTest(app, prod, index)
    
    finishJobs()

    # Convert each app dictionary to a DataFrame.
    for app in enabledApps:
        print("Saving DataFrame for app: " + app)
        df[app] = pd.DataFrame(features[app])
        # Save parameters and results to CSV for optional recovery.
        df[app].to_csv("./tests/" + app + "/dataset.csv")

    # Begin machine learning task.
    # TODO


# TODO: Consider adding code to run random permutations of tests outside of the specific set of tests we have defined. This extra variety can only help training.
# Outline:
# For each parameter:
    # If it is a number:
        # Get the min and max values in the sweep.
        # Pick a random number between min and max to use as the parameter value.
    # Else if it is a string:
        # Pick one of the strings at random.
# Cancel at any time via keystroke.


def main():
    # TODO: Optionally start training from CSV immediately.

    slightAdjustParams()
    #wideAdjustParams()


if __name__ == "__main__":
    main()
