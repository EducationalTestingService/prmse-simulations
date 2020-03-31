"""Utility functions for PRMSE simulatione experiments."""

import itertools

import numpy as np
import pandas as pd
from rsmtool.analyzer import Analyzer
from rsmtool.utils.prmse import prmse_true
from scipy.stats import pearsonr


def get_rater_pairs(rater_ids, num_pairs, seed=1234567890):
    """
    Randomly sample given number of rater pairs from given rater IDs.

    Parameters
    ----------
    rater_ids : list of str
        A list of rater IDs from which we have to randomly sample pairs.
    num_pairs : int
        Number of rater pairs we want to sample.
    seed : int, optional
        The seed for the random number generator that will be
        used to sample the rater pairs.
        Defaults to 1234567890.

    Returns
    -------
    rater_pairs : list of str
        A list containing the required number of randomly sampled
        rater pairs. Each pair is of the form "h_X+h_Y".
    """
    # first we generate all possible rater pair combinations
    all_pairs = [f"{rater1}+{rater2}" for (rater1, rater2)
                 in itertools.combinations(rater_ids, 2)]

    # next we randomly sample as many rater pairs as we need
    prng = np.random.RandomState(seed)
    chosen_pairs = prng.choice(all_pairs, size=num_pairs, replace=False)
    return [pair.split('+') for pair in chosen_pairs]


def compute_agreements_for_rater_pair(df_scores, system_id, rater_id1, rater_id2):
    """
    Evaluate the given system against the given pair of raters.

    This function computes the agreement metrics between the scores
    assigned by the given simulated system (``system_id``) against the scores
    assigned by the two simulated raters ``rater_id1`` and ``rater_id2``.

    The agreement metrics computed are: Pearson's correlation, adjusted R^2,
    quadratically-weighted kappa, and the difference between the human-machine
    Pearson correlation and the human-human Pearson correlation (commonly
    known as "degradation"). All 4 metrics are computed against the scores
    of the first rater in the pair as well as against the average of the scores
    assigned by both raters in the pair.

    Parameters
    ----------
    df_scores : pandas.DataFrame
        The data frame containing the simulated scores.
        This is usually one of the data frames returned
        by the ``simulation.dataset.Dataset.to_frame()``
        method.
    system_id : str
        The ID for the simulated system to be evaluated.
        This must be a column in ``df_scores``.
    rater_id1 : str
        The ID for the first rater in the rater pair
        being used to evaluate the given system.
        This must be a column in ``df_scores``.
    rater_id2 : str
        The ID for the second rater in the rater pair
        being used to evaluate the given system.

    Returns
    -------
    metrics_series : list of pandas.Series
        A list containing two pandas series : the first containing
        the values of the metrics against the average of the two
        rater scores and the second containing the value of the
        metrics against the scores of the first rater. Each series
        contains the following columns:
        1. "r" - the Pearson's correlation between the system score
           and the average and the first rater scores.
        2. "QWK" - the quadratically-weighted kapp between the system score
           and the average and the first rater scores.
        3. "R2" - the R^2 score between the system score
           and the average and the first rater scores.
        4. "degradation" - the difference between the human-machine correlation
           score and the rater1-rater2 correlation score.
        5. "reference" - a column containing whether the metric values were
           computed against the average of the two rater scores (``h1-h2 mean``)
           or the first rater scores (``h1``).
    """
    # compute the inter-rater correlation that we need for degradation
    rater1_rater2_correlation = pearsonr(df_scores[rater_id1], df_scores[rater_id2])[0]

    # we only want these 3 metrics to start with
    chosen_metrics = ['wtkappa', 'corr', 'R2']

    # compute the metrics against the average ot the two rater scores as a series
    mean_metric_values = Analyzer.metrics_helper(df_scores[[rater_id1, rater_id2]].mean(axis=1),
                                                 df_scores[system_id])
    mean_metric_values = mean_metric_values[chosen_metrics]

    # compute the metrics against the first rater as a series
    h1_metric_values = Analyzer.metrics_helper(df_scores[rater_id1], df_scores[system_id])
    h1_metric_values = h1_metric_values[chosen_metrics]

    # compute the degradation values
    mean_metric_values['degradation'] = mean_metric_values['corr'] - rater1_rater2_correlation
    h1_metric_values['degradation'] = h1_metric_values['corr'] - rater1_rater2_correlation

    # add a new column called "reference" indicating whether we used
    # the h1-h2 average score for just the h1 score
    mean_metric_values['reference'] = 'h1-h2 mean'
    h1_metric_values['reference'] = 'h1'

    # rename some of the metrics to have more recognizable names
    mean_metric_values.rename({'wtkappa': 'QWK', 'corr': 'r'}, inplace=True)
    h1_metric_values.rename({'wtkappa': 'QWK', 'corr': 'r'}, inplace=True)

    # return the two metric series
    return [mean_metric_values, h1_metric_values]


def compute_conventional_agreement_metrics(df_scores, system_id, rater_pairs):
    """
    Compute agreement for system against all given rater pairs.

    This function computes the values of conventional metrics of agreement
    between the scores of the given system (``system_id``) against the scores
    assigned by the two simulated raters ``rater_id1`` and ``rater_id2``.

    This function simply calls the ``compute_agreements_for_rater_pair()``
    function for each rater pair function and combines the output. Refer
    to ``compute_agreements_for_rater_pair()`` for more details.

    Parameters
    ----------
    df_scores : pandas.DataFrame
        The data frame containing the simulated scores.
        This is usually one of the data frames returned
        by the ``simulation.dataset.Dataset.to_frame()``
        method.
    system_id : str
        The ID for the simulated system to be evaluated.
        This must be a column in ``df_scores``.
        Description
    rater_pairs : list of lists of str
        A list containing rater pairs against which
        the system is to be evaluated. Each rater
        pair is a list of rater ID, e.g.,
        ``[h_1, h_33]``.

    Returns
    -------
    df_metrics : pandas.DataFrame
        A pandas DataFrames which has the same columns as the series
        returned by ``compute_agreements_for_rater_pair()``.
    """
    # initialize a list that will hold the series
    metrics_for_all_pairs = []

    # iterate over each given rater pair
    for rater_id1, rater_id2 in rater_pairs:

        # call the per-pair function
        metrics_for_this_pair = compute_agreements_for_rater_pair(df_scores,
                                                                  system_id,
                                                                  rater_id1,
                                                                  rater_id2)
        # save the returned lists of serie
        metrics_for_all_pairs.extend(metrics_for_this_pair)

    # create a data frame from the lists of series
    df_metrics = pd.DataFrame(metrics_for_all_pairs)

    return df_metrics


def compute_prmse(df_scores, system_id, rater_pairs):
    """
    Compute the PRMSE score for the system against all given rater pairs.

    This function computes the value of the PRMSE metric between
    the scores of the given system (``system_id``) against the scores
    assigned by the two simulated raters ``rater_id1`` and ``rater_id2``.

    Parameters
    ----------
    df_scores : pandas.DataFrame
        The data frame containing the simulated scores.
        This is usually one of the data frames returned
        by the ``simulation.dataset.Dataset.to_frame()``
        method.
    system_id : str
        The ID for the simulated system to be evaluated.
        This must be a column in ``df_scores``.
        Description
    rater_pairs : list of lists of str
        A list containing rater pairs against which
        the system is to be evaluated. Each rater
        pair is a list of rater ID, e.g.,
        ``[h_1, h_33]``.

    Returns
    -------
    prmse_values : list of float
        A list containing the values for the PRMSE metric
        for each of the given rater pairs.
    """
    # initialize a list that will hold the series
    prmse_for_all_pairs = []

    # iterate over each given rater pair
    for rater_id1, rater_id2 in rater_pairs:

        # call the per-pair function
        prmse_for_this_pair = prmse_true(df_scores[system_id],
                                         df_scores[[rater_id1, rater_id2]])
        # save the returned lists of serie
        prmse_for_all_pairs.append(prmse_for_this_pair)

    return prmse_for_all_pairs
