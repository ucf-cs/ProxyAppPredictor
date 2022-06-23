# ProxyAppPredictor
Generating, testing, and parsing inputs for a  variety of Proxy Apps, both locally and on the Voltrino HPC testbed.

## Dependencies

```bash
pip install wheel numpy pandas sklearn matplotlib
```

## Machine Learning Output RegEx Substitution:

Run the following substitution on `MLoutput.txt` through https://regex101.com/ with the Python flavor:

REGULAR EXPRESSION
```
([A-Za-z0-9 ()]+) ([A-Za-z]+)\n R\^2: ([0-9\-\.]+)\n([0-9\.]+)s 
```
SUBSTITUTION
```
\1\t\3\t\4
```