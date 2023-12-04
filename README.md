# Balancing Summarization and Change Detection in Graph Streams 

## Setup
To build the environment, run the following script: 
```bash
$ conda env create -f=bsc.yaml
```

For calculating the parametric complexity, run the following script in the directory `util`: 
```bash
$ conda install -c conda-forge cxx-compiler
$ cythonize -3 -a -i cython_normterm_discrete.pyx
``` 

## Synthetic Dataset
In the directory `synthetic`, Run the following script: 
```bash
$ python bsc_synthetic.py
```

## TwitterWorldCup2014 Dataset
The TwitterWorldCup2014 dataset is available from the following site: 

[TwitterWorldCup2014 dataset](https://odds.cs.stonybrook.edu/twitterworldcup2014-dataset/)

The datasets `Soccer World Cup 2014 - Ground truth_SpreadSheet.xlsx` and `Twitter_WorldCup_2014_resolved.txt` should be stored in the directory `real/twitter/data/`. 

Run the following script for preprocessing in the directory `real/twitter`:
```bash
$ python makedata_twitter.py
```

Then, run the following script: 
```bash
$ python bsc_twitter.py
```
