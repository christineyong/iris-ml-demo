# Iris Machine Learning Demo
This repository contains code that tests different Machine Learning models on the Iris dataset.

This repository was originally started to showcase how Git can be used for version-controlling a machine learning and data science codebase.

## Prerequisites
Dependencies that should be installed before using this repository:
```
python==3.9
conda (any distribution)
```

## How to use
1. Clone this directory to the desired directory on your machine.
2. `cd` into the repository and install dependencies into a new virtual environment `iris-ml` by running `conda env create -f environment.yml`
3. To train, evaluate and save the best-performing model, run `python3 -m iris_ml.train`. Model artifacts and logs will be grouped by run hash (a hash unique to each run) and saved into the `runs/` directory. (Bug: note that logs currently do not save to the logs file when the model run is initiated using `pytest`)
4. To test, simply run `pytest`. To view `stdout` log output when running `pytest`, run `pytest -o log_cli=true --log-cli-level=10`.
