# Particle Filter SLAM

This projects adapts an existing SLAM implementation (<https://github.com/yashv28/Particle-Filter-SLAM>) to work with a different dataset, fixes a couple issues in the original code and provides insights into the performance of the algorithm.

* Experiments ran on: Windows 10, Intel(R) Core(TM) i7-11370H @ 3.30GHz
* Python 3.12.3

## Configuration

All configurable hyperparameters are provided at the top of `SLAM.py`. Comments in the file provide clarification on what each parameter's function is.

## Dataset

Besides the dataset the original authors supplied, the code has been modified to support the [Rawseeds](http://www.rawseeds.org/home/). The `data` folder contains an archive for the `Bicocca_2009-02-25b` recording session. In order to use this dataset, the archive must be extracted in the data folder (so that the CSV files end up in the `data/Bicocca_2009-02-25b` folder).

# Usage

Run `python SLAM.py` to run the application. Depending on the configuration parameters, you will either see a live output or the simulation will be fully ran first before showing the final map.

## Experiments

The code is setup to provide reproducable experiments with predefined hyperparameters.

- Run `python SLAM.py experiments/experiment0_default.json` (or a different experiment file) to run that specific experiment.
- Run `run_all.bat` to sequentially run all experiments

Running an experiment will output an associated `{experiment}_map.png` file with the map and RMSE graph, and an `{experiment}_stats.txt` file with the hyperparameters and final runtime and RSME values.
