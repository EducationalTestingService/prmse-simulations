# PRMSE Simulations

This repository contains the simulation code underlying the paper *Using PRMSE to evaluate automated scoring systems in the presence of label noise* published at the 15th [ACL BEA workshop](https://sig-edu.org/bea/current) in 2020.

## Getting Started

1. Create a conda environment called by running `conda env create -f environment.yml`. This command might take a few minutes. Please wait for it to finish.

2. Activate this new environment by running `conda activate prmse`.

## Analyses

### Data

The simulated dataset used in the paper is stored as a combination of `.csv` files under `data`:

- [`scores.csv`](data/scores.csv) - simulated human, machine, and true scores

- [`rater_metadata.csv`](data/rater_metadata.csv) - information about each simulated "human" rater

- [`system_metadata.csv`](data/system_metadata.csv) - information about each simulated "system".

These `.csv` files are provided for reference and will *not* be overwritten by the notebook. 

The same dataset is also stored as [`default.dataset`](data/default.dataset) file: a serialized instance of the `Dataset` class used in all simulations (see [notes](#important-notes) section below ). This file *will* be overwritten if you make changes to the notebooks or to the settings. 

### Simulations

The code for the simulations is divided into a set of Jupyter notebooks under the `notebooks` directory.

1. [`making_a_dataset.ipynb`](notebooks/making_a_dataset.ipynb). This is the notebook used to create a simulated dataset using the dataset parameters stored in [`notebooks/dataset.json`](notebooks/dataset.json). In addition to creating the dataset, it also contains some preliminary analyses on the dataset to make sure that it behaves as expected. This notebook serializes the dataset and saves it under [`data/default.detaset`](data/default.dataset). This serialized dataset file is then used by the subsequent notebooks to load the dataset. Therefore, changing the parameters in `dataset.json` and re-running this notebook will change the results of the analyses in the other notebooks.

2. [`multiple_raters_true_score.ipynb`](notebooks/multiple_raters_true_score.ipynb).  In this notebook, we explore the impact of using a larger number of human raters in the evaluation process. More specifically, we show that as we use more and more human raters, the average of the scores assigned by said raters approaches the true score. In addition, we show that when evaluating a given automated system against an increasing number of human raters, the values of the conventional agreement metrics approach values that would be computed if that same system were to be evaluated against the true score.

3. [`metric_stability.ipynb`](notebooks/metric_stability.ipynb). In this notebook, we compare the stability of conventional agreement metrics such as Pearson's correlation, quadratically-weighted kappa, mean squared error, and R^2 to that of the proposed PRMSE metric. We do this by showing that the usual agreement metrics can give very different results depending on the pair of human raters that are used as the reference against which the system score is evaluated. However, the PRMSE metric yields stable evaluation results across different pairs of human raters.

4. [`ranking_multiple_systems.ipynb`](notebooks/ranking_multiple_systems.ipynb). In this notebook, we explore how to rank multiple automated scoring systems. Specifically, we consider the situation where we have scores from multiple different automated scoring systems, each with different levels of performance.  We evaluate these systems against the same as well as different pairs of raters and show that while all metrics can rank the systems accurately when using a single rater pair for evaluation, only PRMSE can do the same when a different rater pair is used for every system.

5. [`prmse_and_double_scoring.ipynb`](notebooks/prmse_and_double_scoring.ipynb). In this notebook, we explore the impact of the number of double-scored responses on PRMSE. We know that in order to compute PRMSE, we need at least some of the responses to have scores from two human raters. However, it may not be practical to have every single response double-scored. In this notebook, we examine how PRMSE depends on the number of double-scored responses that may be available in the dataset.

### Running your own simulations

If you are interested in running your own PRMSE simulations, you need to:

1. Edit the [`dataset.json`](notebooks/dataset.json) file to change any of the following dataset-specific settings:
    - the number of responses in the dataset (`num_responses`)
    - the distribution underlying the true scores and the total number of score points (`true_score_mean`, `true_score_sd`, `min_score`, and `max_score`)
    - new categories of simulated human raters and automated systems (`rater_categories`, `rater_rho_per_category`, `system_categories`, and `system_r2_per_category`)
    - the number of simulated rater and/or systems per category (`num_raters_per_category` and `num_systems_per_category`)

2. Run the [`making_a_dataset.ipynb`](notebooks/making_a_dataset.ipynb) notebook to create and save your new dataset instance as `data/default.dataset`.

3. Edit the [`settings.json`](notebooks/settings.json) file to change any of the following notebook-specific simulation settings:
    - `double_scored_percentages` : the percentages of double-scored responses that are simulated in [`prmse_and_double_scoring.ipynb`](notebooks/prmse_and_double_scoring.ipynb).
    - `key_steps_n_raters` : the number of raters included in the cumulative calculations in [`multiple_raters_true_score.ipynb`](notebooks/multiple_raters_true_score.ipynb).
    - `rater_pairs_per_category` : the pre-determined number of rater pairs per category used in  [`metric_stability.ipynb`](notebooks/metric_stability.ipynb), [`ranking_multiple_systems.ipynb`](notebooks/ranking_multiple_systems.ipynb), and [`prmse_and_double_scoring.ipynb`](notebooks/prmse_and_double_scoring.ipynb).
    - `sample_system` : the simulated automated scoring system chosen as the source of automated scores in [`multiple_raters_true_score.ipynb`](notebooks/multiple_raters_true_score.ipynb) and [`metric_stability.ipynb`](notebooks/metric_stability.ipynb).

3. Run the notebooks to see how PRMSE performs for your simulation settings.

### Important Notes

1. Note that the structure and order of the notebooks does not necessarily follow the order of analyses in the paper. For example, in the paper we first show the gaps in traditional metrics and then demonstrate that PRMSE can help address those. However, in the notebooks, it is more efficient to keep the analyses with and without PRMSE in the same notebook as long as they use the same data. 

2. For efficiency and readability reasons, a lot of code shared by the notebooks is factored out into a package called `simulation` found under [`notebooks/simulation`](notebooks/simulation). This package contains two main Python files:

    - [`simulation/dataset.py`](notebooks/simulation/dataset.py). This module contains the main ``Dataset`` class representing the simulated dataset underlying all of the PRMSE simulations.

    - [`simulation/utils.py`](notebooks/simulation/dataset.py). This module contains several utility functions needed for the various simulations in the notebooks.

3. Running the [`making_a_dataset.ipynb`](notebooks/making_a_dataset.ipynb) also saves three CSV files, one for each of the data frames that can be obtained by calling the `to_frames()` method on the dataset instance saved in `data/default.dataset`. Between themsleves, these 3 CSV files contain all of the simulated scores as well as the rater and system metadata. For a detailed description of each data frame, see the docstring for the `to_frames()` method of the [`Dataset`](notebooks/simulation/dataset.py) class. We make these CSV files available under `data` so that they can be examined and modified in other programs such as Excel and R. However, making changes to these CSV files will _not_ affect any analyses in any of the other notebooks as they use the `data/default.dataset` file and _not_ these CSV files.

## License

The code and data in this repository is released under the MIT license. 
