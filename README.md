# ProxyAppPredictor
Generating, testing, and parsing inputs for a  variety of Proxy Apps, both locally and on HPC systems.

## Associated paper

[Evaluating HPC Job Run Time Predictions Using Application Input Parameters](https://dl.acm.org/doi/10.1145/3583678.3596893)

## Dependencies

```bash
pip install wheel numpy pandas sklearn matplotlib
```

## How to use

Note that scripts are set up for use on Eclipse at Sandia. Hard-coded environments, accounts, queue names, etc. will need to be adjusted to suit your specific HPC system.

### Running tests

`runTesting.sh` loads the appropriate Python module to grant access to scikit-learn and other Python requirements, then generates and runs tests at random until cancelled. Each test gets its own directory, with input and output files, for troubleshooting purposes. Application input parameters and run time are saved to one CSV per application. These CSVs can then be used to train models and evaluate prediction performance.

```bash
bash runTesting.sh
```

### Evaluating tests

`HPCTesting.sh` leverages HPC machines to evaluate each combination of application and model in parallel. Each test will train a model, then evaluate its performance. Output includes prediction accuracy and training time metrics as well as graphs.

```bash
bash HPCTesting.sh
```

## Data set

The data set used in the published paper can be found in the [`tests/`](tests/) directory.

# Notes

Evaluation data requires some processing to make it usable in tables.

## Machine learning output RegEx substitution:

Run the following substitutions on `MLoutput.txt` through https://regex101.com/ with the Python flavor:

### Organize important data into tidy rows for use by Excel:

REGULAR EXPRESSION
```
([A-Za-z0-9 ()]+) ([A-Za-z\-]+)\n R\^2: ([0-9\-\.]+|nan)\n([0-9\.]+)s 
```
SUBSTITUTION
```
\1\t\2\t\3\t\4
```

### Remove lines without tabs:

REGULAR EXPRESSION
```
^[^	]*$
```
SUBSTITUTION
```

```

### Remove newlines:

REGULAR EXPRESSION
```
\n\n
```
SUBSTITUTION
```
\n
```

## View SLURM queue status:

You can monitor your jobs to keep track of queued jobs and node usage.

```bash
watch "squeue -u kmlamar && squeue -u kmlamar | grep -c kmlamar && sinfo"
```