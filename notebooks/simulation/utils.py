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


def compute_agreement_one_system_one_rater_pair(df_scores,
                                                system_id,
                                                rater_id1,
                                                rater_id2,
                                                include_mean=False):
    """
    Evaluate the given system against the given pair of raters.

    This function computes the agreement metrics between the scores
    assigned by the given simulated system (``system_id``) against the scores
    assigned by the two simulated raters ``rater_id1`` and ``rater_id2``.

    The agreement metrics computed are: Pearson's correlation, adjusted R^2,
    quadratically-weighted kappa, and the difference between the human-human
    Pearson correlation and the human-machine Pearson correlation (commonly
    known as "degradation"). All 4 metrics are computed against the scores
    of the first rater in the pair and, if ``include_mean`` is ``True``, also
    against the average of the scores assigned by both raters in the pair.

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
    include_mean : bool, optional
        If set to ``True``, also include the metric values
        computed against the average of the scores assigned
        by both raters in the given pair.

    Returns
    -------
    metrics_series : list of pandas.Series
        A list containing 1 or 2 pandas series depending on the value
        of ``include_mean``. If it is ``True``, this list contains
        two series:  the first containing the values of the metrics
        against the average of the two rater scores and the second
        containing the value of the metrics against the scores of
        the first rater. If ``include_mean`` is ``False``, this list
        only contains a single series - that containing the metric
        values against the scores of the first rater. Any series
        returned contain the following columns:
        1. "r" - the Pearson's correlation between the system score
           and the average and the first rater scores.
        2. "QWK" - the quadratically-weighted kapp between the system score
           and the average and the first rater scores.
        3. "R2" - the R^2 score between the system score
           and the average and the first rater scores.
        4. "degradation" - the difference between the human-human correlation
           score and the system's correlation score.
        5. "reference" - a column containing whether the metric values were
           computed against the average of the two rater scores (``h1-h2 mean``)
           or the first rater scores (``h1``).
    """
    # compute the inter-rater correlation that we need for degradation
    rater1_rater2_correlation = pearsonr(df_scores[rater_id1], df_scores[rater_id2])[0]

    # we only want these 3 metrics to start with
    chosen_metrics = ['wtkappa', 'corr', 'R2']

    # compute the metrics against the first rater as a series
    h1_metric_values = Analyzer.metrics_helper(df_scores[rater_id1], df_scores[system_id])
    h1_metric_values = h1_metric_values[chosen_metrics]

    # compute the degradation values
    h1_metric_values['degradation'] = rater1_rater2_correlation - h1_metric_values['corr']

    # add a new column called "reference" indicating whether we used
    # the h1-h2 average score for just the h1 score
    h1_metric_values['reference'] = 'h1'

    # rename some of the metrics to have more recognizable names
    h1_metric_values.rename({'wtkappa': 'QWK', 'corr': 'r'}, inplace=True)

    # compute the metrics against the average ot the two rater scores
    # as a series if it was requested
    if include_mean:
        mean_metric_values = Analyzer.metrics_helper(df_scores[[rater_id1, rater_id2]].mean(axis=1),
                                                     df_scores[system_id])
        mean_metric_values = mean_metric_values[chosen_metrics]

        mean_metric_values['degradation'] = rater1_rater2_correlation - mean_metric_values['corr']
        mean_metric_values['reference'] = 'h1-h2 mean'
        mean_metric_values.rename({'wtkappa': 'QWK', 'corr': 'r'}, inplace=True)

    # return the right number of metric series
    ans = [mean_metric_values, h1_metric_values] if include_mean else [h1_metric_values]
    return ans


def compute_agreement_one_system_multiple_rater_pairs(df_scores,
                                                      system_id,
                                                      rater_pairs,
                                                      include_mean=False):
    """
    Compute agreement for system against all given rater pairs.

    This function computes the values of conventional metrics of agreement
    between the scores of the given system (``system_id``) against the scores
    assigned by the two simulated raters ``rater_id1`` and ``rater_id2``.

    This function simply calls the ``compute_agreement_one_system_one_rater_pair()``
    function for each rater pair and combines the output. Refer
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
    include_mean : bool, optional
        If set to ``True``, also include the metric values
        computed against the average of the scores assigned
        by both raters in the given pair.

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
        metrics_for_this_pair = compute_agreement_one_system_one_rater_pair(df_scores,
                                                                            system_id,
                                                                            rater_id1,
                                                                            rater_id2,
                                                                            include_mean=include_mean)
        # save the returned lists of serie
        metrics_for_all_pairs.extend(metrics_for_this_pair)

    # create a data frame from the lists of series
    df_metrics = pd.DataFrame(metrics_for_all_pairs)

    return df_metrics


def compute_agreement_multiple_systems_one_rater_pair(df_scores,
                                                      system_ids,
                                                      rater_id1,
                                                      rater_id2,
                                                      include_mean=False):
    """
    Compute agreement for given systems against the given rater pair.

    This function computes the values of conventional metrics of agreement
    between the scores of the given list of systems (with IDs ``system_ids``)
    against the scores assigned by the two simulated raters ``rater_id1``
    and ``rater_id2``.

    This function simply calls the ``compute_agreement_one_system_one_rater_pair()``
    function for each system and combines the output. Refer to
    ``compute_agreements_for_rater_pair()`` for more details.

    Parameters
    ----------
    df_scores : pandas.DataFrame
        The data frame containing the simulated scores.
        This is usually one of the data frames returned
        by the ``simulation.dataset.Dataset.to_frame()``
        method.
    system_ids : list of str
        The list of IDs of the simulated systems to be evaluated.
        Each ID must be a column in ``df_scores``.
    rater_id1 : str
        The ID for the first rater in the rater pair
        being used to evaluate the given system.
        This must be a column in ``df_scores``.
    rater_id2 : str
        The ID for the second rater in the rater pair
        being used to evaluate the given system.

    Returns
    -------
    data frame
        Description
    """
    # initialize an empty list we will use to save each system ID's results
    metrics = []

    # iterate over each system ID
    for system_id in system_ids:

        # compute the metric series for this system ID against the rater pair
        metric_series = compute_agreement_one_system_one_rater_pair(df_scores,
                                                                    system_id,
                                                                    rater_id1,
                                                                    rater_id2,
                                                                    include_mean=include_mean)
        # save the current system ID in the same series
        for series in metric_series:
            series['system_id'] = system_id

        # save the series in the list
        metrics.extend(metric_series)

    # convert the list of series into a data frame and return
    return pd.DataFrame(metrics)


def compute_prmse_one_system_multiple_rater_pairs(df_scores, system_id, rater_pairs):
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
        # save the returned lists of series
        prmse_for_all_pairs.append(prmse_for_this_pair)

    return prmse_for_all_pairs


def compute_cumulative_mean_for_raters(df_scores, rater_ids):
    """
    Compute cumulative average of given rater's scores.

    This function computes the cumulative average of scores
    assigned by the raters identified by the given list of
    ``rater_ids``. The cumulative average is computed by
    adding one rater at a time.

    Parameters
    ----------
    df_scores : pandas DataFrame
        The data frame containing the simulated scores.
        This is usually one of the data frames returned
        by the ``simulation.dataset.Dataset.to_frame()``
        method.
    rater_ids : numpy.ndarray
        An array of simulated rater IDs whose scores want
        to compute the cumulative average over.

    Returns
    -------
    df_cumulative_mean_scores : pandas.DataFrame
        A data frame containing the cumulative average rater scores.
        It has ``len(df_scores)`` rows and ``len(rater_ids`` columns.
        Each of the columns is named ``N=n``, where n is the number of
        raters included in computing the cumulative average.
    """
    df_rater_scores = df_scores[rater_ids]
    df_cumulative_average_scores = df_rater_scores.expanding(1, axis=1).mean()
    df_cumulative_average_scores.columns = [f"N={num_raters+1}" for num_raters in range(len(rater_ids))]
    return df_cumulative_average_scores


def compute_ranks_from_metrics(df_metrics):
    """
    Compute ranks given metric values for systems.

    This function computes ranks for a list of systems
    according to different metrics present in the given
    data frame.

    Parameters
    ----------
    df_metrics : pandas.DataFrame
        A data frame with one row for each system to be ranked and
        the following columns:
        1. "system_id" : the ID of the system
        2. "system_category" : the performance category that the system belongs to.
        3. At least one other column containing values for a metric with the name
       of the metric being the column name. For example, "QWK" or "PRMSE" etc. 

    Returns
    -------
    df_metric_ranks : pandas.DataFrame
        A data frame with the same number of rows and columns as the input
        data frame except that the values in each "metric" column are now the
        ranks of the systems rather than the metric values themselves.
    """
    # first we set our indices to be the system IDs and categories so that they are
    # retained when we compute the ranks for the systems
    df_metrics_for_ranks = df_metrics.set_index(['system_category', 'system_id'])

    # if degradation is one of the metrics, multiply it by -1 to make it behave'
    # like other metrics for ranking purposes
    if "degradation" in df_metrics_for_ranks.columns:
        df_metrics_for_ranks['degradation'] = -1 * df_metrics_for_ranks['degradation']

    # compute the ranks in descending order since lower ranks are better;
    # also reset the indices so that we get the IDs and categories back
    df_metric_ranks = df_metrics_for_ranks.rank(ascending=False).reset_index()

    # return the data frame
    return df_metric_ranks
