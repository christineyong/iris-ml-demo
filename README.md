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
3. To train, evaluate and save the best-performing model, run `python -m iris_ml.model`. Models will be saved to `models/`.
4. To test, run `pytest`.
