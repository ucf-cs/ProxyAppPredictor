# ProxyAppPredictor
Generating, testing, and parsing inputs for a  variety of Proxy Apps, both locally and on HPC systems.

## Dependencies

```bash
pip install wheel numpy pandas sklearn matplotlib
```

## Machine Learning Output RegEx Substitution:

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

```bash
watch "squeue -u kmlamar && squeue -u kmlamar | grep -c kmlamar && sinfo"
```