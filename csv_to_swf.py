import csv
import os
import sys
import concurrent.futures

# An ignore list of parameters that aren't real extras and shouldn't be included.
ignore_list = ["error","testNum","timeTaken"]
#ignore_list += ["nodes","tasks"]
ignore_list += ["JobID","Submit","Start","ElapsedRaw","NNodes","AveCPU",
                "MaxVMSize","ReqNodes","TimelimitRaw","ReqMem","State","UID",
                "GID","Partition"]
# Add any artificially created fields here too.
ignore_list += ["Wait"]

def csv_to_swf(path):
    # Prevent blank values for irrelevant fields.
    def autofill(str):
        return ("-1" if str == "" else str)

    swf_map = {}
    swf_map["line_count"] = 0
    swf_map["start_time"] = 0
    swf_map["end_time"] = 0

    # Organized as follows:
    # job_dict[app_name][test_num][field_name] = field_value
    job_dict = {}
    # The job data from the sacct command, cached for performance.
    sacct_data = get_sacct_data()

#futures = []
#with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
    print(os.listdir(path))
    for filename in os.listdir(path):
        if not filename.endswith(".csv"):
            continue
        app = filename[0:len(filename)-len("dataset.csv")]
        job_dict[app] = {}
        with open(os.path.join(path, filename), 'r') as csv_file:
            # Prepare the CSV reader.
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            # CSV header data used for organization purposes.
            header_read = False
            header = []
            # Read in each row (job) from the CSV.
            for row in reader:
                swf_map["line_count"] += 1
                # Read the header for the first row.
                if not header_read:
                    header_read = True
                    header = row
                    continue

                #futures.append(executor.submit(handle_row, row, app, job_dict, header, swf_map, sacct_data, path))
                handle_row(row, app, job_dict, header, swf_map, sacct_data, path)
        print("Finished queuing all lines from "+filename)
    # for future in futures:
    #     future.result()
    print("Finished reading all lines")


    contents = ('; Version: 2.2\n'
                '; Computer: Intel Haswell/KNL hybrid\n'
                '; Installation: Sandia National Labs, Voltrino\n'
                '; Acknowledge: Kenneth Lamar\n'
                '; Information: https://github.com/KennethLamar/ProxyAppPredictor/\n'
                '; Conversion: Kenneth Lamar\n'
                '; MaxJobs: {line_count}\n'
                '; MaxRecords: {line_count}\n'
                '; Preemption: No\n'
                '; UnixStartTime: -1\n' # NOTE: Ensure this is generated using Unix "date" command.
                '; TimeZone: -18000\n'
                '; TimeZoneString: US/Eastern\n'
                '; StartTime: {start_time}\n'
                '; EndTime: {end_time}\n'
                '; MaxNodes: 4 (More on KNL)\n'
                '; MaxProcs: 32\n'
                '; MaxRuntime: 86400\n'
                '; MaxMemory: 128G\n'
                '; Note: uses the FCFS scheduler\n'
                '; Queues: Distinguish which type of CPUs to run on\n') \
                .format_map(swf_map)
    # NOTE: Each application gets a unique number from 1 upward.
    # This also dictates what the extra fields mean.
    app_id = 0
    app_ids = {}
    # Iterate over every job type.
    for app_name in job_dict:
        app_id += 1
        app_ids[app_name] = app_id
        # Every job will make a note of what its extra fields mean.
        contents += '; params ' + str(app_id) + ":"
        # Iterate over every parameter.
        job_id_key = list(job_dict[app_name])[0]
        for param in job_dict[app_name][job_id_key]:
            # Report the extra field.
            contents += str(param)+', '
        contents += '\n'
    job_id = 0
    # Iterate again, this time printing out all fields.
    for app_name in job_dict:
        for test_num in job_dict[app_name]:
            job_id += 1
            try:
                # Job Number
                line = autofill(str(job_id)) + " "
                # Submit Time
                line += autofill(str(job_dict[app_name][test_num]["Submit"])) + " "
                # Wait Time
                line += autofill(str(job_dict[app_name][test_num]["Start"])) + " "
                # Run Time
                line += autofill(str(job_dict[app_name][test_num]["ElapsedRaw"])) + " "
                # Number of Allocated Processors
                line += autofill(str(job_dict[app_name][test_num]["NNodes"])) + " "
                # Average CPU Time Used
                line += autofill(str(job_dict[app_name][test_num]["AveCPU"])) + " "
                # Used Memory
                line += autofill(str(job_dict[app_name][test_num]["MaxVMSize"])) + " "
                # Requested Number of Processors
                line += autofill(str(job_dict[app_name][test_num]["ReqNodes"])) + " "
                # Requested Time
                line += autofill(str(job_dict[app_name][test_num]["TimelimitRaw"])) + " "
                # Requested Memory
                line += autofill(str(job_dict[app_name][test_num]["ReqMem"])) + " "
                # Status
                line += autofill(str(job_dict[app_name][test_num]["State"])) + " "
                # User ID
                line += autofill(str(job_dict[app_name][test_num]["UID"])) + " "
                # Group ID
                line += autofill(str(job_dict[app_name][test_num]["GID"])) + " "
                # Executable (Application) Number
                line += autofill(str(app_ids[app_name])) + " "
                # Queue Number
                line += autofill(str(-1)) + " "
                # Partition Number
                line += autofill(str(job_dict[app_name][test_num]["Partition"])) + " "
                # Preceding Job Number
                line += autofill(str(-1)) + " "
                # Think Time from Preceding Job
                line += autofill(str(-1)) + " "
                # Iterate over every extra parameter.
                for param in job_dict[app_name][test_num]:
                    if param in ignore_list:
                        continue
                    value = str(job_dict[app_name][test_num][param])
                    if value == "":
                        print("No value associated with "+str(app_name)+" "+str(test_num)+" "+str(param))
                    line += value + " "
                line += "\n"
                contents += line
            except KeyError as e:
                # Skip jobs that are missing data.
                print(e)
                pass
    return contents


def handle_row(row, app, job_dict, header, swf_map, sacct_data, path):
    # Read each element into the dictionary.
    # Get testNum.
    test_num = row[header.index("testNum")]
    job_dict[app][test_num] = {}
    # Read in each field associated with this test.
    for index, cell in enumerate(row):
        job_dict[app][test_num][header[index]] = cell

    # Retrieve the jobid associated with this job from Voltrino logs.
    jobid = 0
    # Open the associated folder.
    test_path = "E:/tests/tests/" + app + "/" + str(test_num).zfill(10)
    for filename in os.listdir(test_path):
        # Find the associated job ID.
        if "slurm-" in filename:
            jobid = int(filename[len("slurm-"):len(filename)-len(".out")])
            break
    job_dict[app][test_num]["JobID"] = jobid

    # Assign the appropriate sacct data.
    job_id = str(job_dict[app][test_num]["JobID"])
    job_dict[app][test_num].update(sacct_data[job_id])

    # Update swf data as needed.
    if swf_map["start_time"] > job_dict[app][test_num]["Start"]:
        swf_map["start_time"] = job_dict[app][test_num]["Submit"]
    if swf_map["end_time"] < job_dict[app][test_num]["Start"] \
                        + job_dict[app][test_num]["ElapsedRaw"]:
        swf_map["end_time"] = job_dict[app][test_num]["Start"] \
                            + job_dict[app][test_num]["ElapsedRaw"]
    #print("Handled row " + str(test_num))


# Extract remaining parameters by running sacct for this job.
def get_sacct_data():
    sacct_data = {}
    # Partition mapping.
    partitions = {}
    partition_idx = 1
    # sacct = subprocess.run("sacct -j "+jobid+" --format=JobID,Submit,Start,ElapsedRaw,NNodes,AveCPU,MaxVMSize,ReqNodes,TimelimitRaw,ReqMem,State,UID,GID,Partition", stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,check=False,encoding='utf-8').stdout
    # TEMP: Fallback file.
    with open("C:/Users/MarioMan/Desktop/kendump.txt", 'r') as sacct_file:
        # Prepare the CSV reader.
        reader = csv.reader(sacct_file, delimiter='|', quotechar='"')
        # CSV header data used for organization purposes.
        header_read = False
        header = []
        # Read in each row (job) from the file.
        for row in reader:
            # Read the header for the first row.
            if not header_read:
                header_read = True
                header = row
                continue
            row_data = {}
            # Read in each field associated with this test.
            for index, cell in enumerate(row):
                # Convert numeric types.
                try:
                    # If it can be a float, make it a float.
                    cell = float(cell)
                    if cell - int(cell) == 0.0:
                        cell = int(cell)
                except ValueError:
                    pass
                row_data[header[index]] = cell

            # Make adjustments to required formatting.
            # Start(convert to wait time)
            row_data["Wait"] = row_data["Start"] - row_data["Submit"]
            # TimelimitRaw (convert from mins to secs)
            row_data["TimelimitRaw"] = row_data["TimelimitRaw"] * 60
            # ElapsedRaw (convert from mins to secs)?
            # TODO: ReqMem (convert based on units (M, g, etc.))
            # NOTE: Always seems to report "0n".
            # DEBUG: Make it -1 for all jobs.
            row_data["ReqMem"] = -1
            # Convert to numbers, starting at 1.
            if not row_data["Partition"] in partitions:
                partitions[row_data["Partition"]] = partition_idx
                partition_idx += 1
            row_data["Partition"] = partitions[row_data["Partition"]]

            # State (Convert using state codes)
            # Completed (1), Failed (0), or cancelled (5)
            # https://slurm.schedmd.com/sacct.html
            state_codes = { "BF": 0,
                            "BOOT_FAIL": 0,
                            "CA": 5,
                            "CANCELLED": 5,
                            "CD": 1,
                            "COMPLETED": 1,
                            "DL": 0,
                            "DEADLINE": 0,
                            "F": 0,
                            "FAILED": 0,
                            "NF": 0,
                            "NODE_FAIL": 0,
                            "OOM": 0,
                            "OUT_OF_MEMORY": 0,
                            "PD": 0,
                            "PENDING": 0,
                            "PR": 0,
                            "PREEMPTED": 0,
                            "R": 0,
                            "RUNNING": 0,
                            "RQ": 0,
                            "REQUEUED": 0,
                            "RS": 0,
                            "RESIZING": 0,
                            "RV": 0,
                            "REVOKED": 0,
                            "S": 0,
                            "SUSPENDED": 0,
                            "TO": 0,
                            "TIMEOUT": 0}
            state = str(row_data["State"]).split()[0].upper()
            row_data["State"] = state_codes[state]

            # Add the row data to the overall sacct dictionary.
            sacct_data[row[header.index("JobID")]] = {}
            sacct_data[row[header.index("JobID")]].update(row_data)
    print("Finished parsing sacct data")
    return sacct_data


def main():
    path = str(sys.argv[1])
    print(path)
    contents = csv_to_swf(path)
    with open(os.path.join(path, "output.swf"), 'w') as swf_file:
        swf_file.write(contents)

if __name__ == "__main__":
    main()
