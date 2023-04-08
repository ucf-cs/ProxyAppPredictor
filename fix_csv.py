""" ProxyAppPredictor generated malformed outputs in some CSVs. 
This script corrects those errors, as the data is still present in surrounding
files.
"""

import csv
import glob
import os
import re

DIR = ".\\**"
# DIR = "/pscratch/kmlamar/**"
PATH = os.path.join(DIR, "*.csv")


# For each CSV file in the directory.
for file in glob.glob(PATH, recursive=True):
    # Don't fix files that are already fixed.
    if "Fixed" in file:
        continue
    # Open the file.
    with open(file, "r", encoding="utf-8") as csv_file:
        # Get the app name.
        app_name = os.path.split(file)[-1][0:-len("dataset.csv")]
        # Prepare the CSV reader.
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        # Read the CSV into a 2D list.
        csv_data = []
        for row in reader:
            csv_data.append([])
            for cell in row:
                csv_data[-1].append(cell)
    # If the time taken is missing from the data.
    if "timeTaken" not in csv_data[0]:
        csv_data[0].append("timeTaken")
        # Find and add the associated time to each test.
        for row in csv_data[1:]:
            # Get the test number.
            test_num = 0 if row[0] == "" else int(row[0])
            # Find the output file.
            slurm_file_glob = os.path.join(
                file[:-len("dataset.csv")], str(test_num).zfill(10), "slurm-*.out")
            slurm_file_path = glob.glob(slurm_file_glob)
            if len(slurm_file_path) < 1:
                print("Missing SLURM output: " + str(slurm_file_path))
                continue
            if len(slurm_file_path) > 1:
                print("Too many SLURM outputs: " + str(slurm_file_path))
                exit()
            # DEBUG
            # print("Continuing using: " + str(slurm_file_path))
            # Open the associated SLURM output.
            with open(slurm_file_path[0], "r", encoding="utf-8") as slurm_file:
                slurm_lines = slurm_file.read().split("\n")
                # Find the line containing the time taken.
                found = False
                for slurm_line in slurm_lines:
                    # Add the time taken to the corrected output.
                    for prefix in ["timeTaken = ", "The DiffOfTime = "]:
                        if slurm_line.startswith(prefix):
                            row.append(slurm_line[len(prefix):])
                            found = True
                            break
                assert (found)
    # Check to see if the timeTaken and error are reversed.
    # The error should always be last.
    # Start with the header.
    if csv_data[0][-1] == "timeTaken":
        assert (csv_data[0][-2] == "error")
        # Swap the columns.
        csv_data[0][-1], csv_data[0][-2] = csv_data[0][-2], csv_data[0][-1]
    # Now check the lines.
    for row in csv_data[1:]:
        if row[-1].isdigit():
            # Swap the columns.
            row[-1], row[-2] = row[-2], row[-1]
    # Save the corrected file.
    with open(file[0:-len(".csv")] + "Fixed" + ".csv", "w", encoding="utf-8",
              newline="\n") as output_file:
        writer = csv.writer(output_file, delimiter=",", quotechar='"')
        for row in csv_data:
            writer.writerow(row)
