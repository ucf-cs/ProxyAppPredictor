# ProxyAppPredictor
Generating, testing, and parsing inputs for a  variety of Proxy Apps, both locally and on the Voltrino HPC testbed.

## Dependencies

```bash
pip install wheel numpy pandas sklearn matplotlib
```

## Machine Learning Output RegEx Substitution:

Run the following substitution on `MLoutput.txt` through https://regex101.com/ with the Python flavor:

TEST STRING
```

 R\^2: ([0-9\-\.]+)
 RMSE: ([0-9\-\.]+)
 MAE: ([0-9\-\.]+)
 MedAE: ([0-9\-\.]+)
 MAE%: ([0-9\-\.]+)
```
SUBSTITUTION
```
\t\1
```