"""
Module containing the Dataset class needed for PRMSE simulations.

:author: Nitin Madnani
:author: Anastassia Loukina
:organization: ETS
:date: March 2020
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Dataset:
    """
    Class encapsulating a single simulated dataset.

    A class encapsulating a given simulated dataset as defined
    by the number of responses, the number and type of human raters,
    and the number and type of automated scoring systems, and other
    attributes.
    """

    def __init__(self,
                 num_responses=10000,
                 rater_categories=['low', 'moderate', 'average', 'high'],
                 system_categories=['poor', 'low', 'medium', 'high', 'perfect'],
                 num_raters_per_category=50,
                 num_systems_per_category=5,
                 rater_rho_per_category=[0.4, 0.55, 0.65, 0.8],
                 system_r2_per_category=[0, 0.4, 0.65, 0.8, 0.99],
                 min_score=1,
                 max_score=6,
                 true_score_mean=3.844,
                 true_score_sd=0.74):
        """
        Create ``Simulation`` instance based on given settings.

        Parameters
        ----------
        num_responses : int
            The total number of responses in this simulated dataset.
            Defaults to 10000.
        rater_categories : list of str
            A list of string labels defining the possible rater
            categories; a rater category is defined by the
            inter-rater agreement in that category.
            Defaults to ``['low', 'moderate', 'average', 'high']``.
        system_categories : list of str
            A list of string labels defining the possible automated scoring
            system categories; a system category is defined by the agreement
            of that system's predictions with the true scores.
            Defaults to ``['poor', 'low', 'medium', 'high', 'perfect']``.
        num_raters_per_category : int
            An integer indicating the number of raters we want
            to simulate in each rater category.
            Defaults to 50.
        num_systems_per_category : int
            An integer indicating the number of scoring systems
            we want to simulate in each system category.
            Defaults to 5.
        rater_rho_per_category : list of float
            A list of pearson (rho) values that define each rater category.
            The first rater category in ``rater_categories`` corresponds
            to the first rho value in this list.
            Defaults to ``[0.4, 0.55, 0.65, 0.8]``.
        system_r2_per_category : list of float
            A list of R^2 values that define each system category.
            The first system category in ``system_categories`` corresponds
            to the first R^2 value in this list.
            Defaults to ``[0, 0.4, 0.65, 0.8, 0.99]``.
        min_score : int
            The lowest human score in this simulated dataset.
            Defaults to 1.
        max_score : int
            The highest human score in this simulated dataset.
            Defaults to 6.
        true_score_mean : float
            The desired mean we want for the simulated gold standard/true
            scores.
            Defaults to 3.844 based on a real dataset.
        true_score_sd : float
            The desired standard deviation we want for the simulated
            gold standard/true scores.
            Defaults to 0.74 based on a real dataset.
        """
        self.num_responses = num_responses

        self.rater_categories = rater_categories
        self.rater_rho_per_category = rater_rho_per_category
        self.num_raters_per_category = num_raters_per_category

        self.system_categories = system_categories
        self.system_r2_per_category = system_r2_per_category
        self.num_systems_per_category = num_systems_per_category

        self.min_score = min_score
        self.max_score = max_score
        self.true_score_mean = true_score_mean
        self.true_score_sd = true_score_sd

        # these attributes are initialized as empty for now
        self._true_scores = None
        self._rater_scores = []
        self._rater_metadata = []
        self._system_scores = []
        self._system_metadata = []

    @classmethod
    def from_dict(cls, argdict):
        """Create a ``Dataset`` instance from the given dictionary."""
        return cls(**argdict)

    @classmethod
    def from_file(cls, dataset_path):
        """Load ``Dataset`` instance from disk."""
        dataset = joblib.load(dataset_path)
        return dataset

    def save(self, dataset_path):
        """Save ``Dataset`` instance to disk."""
        # create the directory if it doesn't exist
        dataset_dir = Path(dataset_path).parent
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True)
        # write out the dataset to disk
        joblib.dump(self, dataset_path)

    def save_frames(self, output_dir):
        """Save the frames obtained via ``to_frames()`` to disk."""
        # create the directory if it doesn't exist
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # get the 3 data frames representing the dataset
        (df_scores,
         df_rater_metadata,
         df_system_metadata) = self.to_frames()

        # write out each of the frames to disk
        df_scores.to_csv(output_dir / 'scores.csv', index=False)
        df_rater_metadata.to_csv(output_dir / 'rater_metadata.csv', index=False)
        df_system_metadata.to_csv(output_dir / 'system_metadata.csv', index=False)

    def __str__(self):
        """Return a string representation of Dataset."""
        ans = "Dataset ("
        ans += f"{self.num_responses} responses, "
        ans += f"scores in [{self.min_score}, {self.max_score}], "
        ans += f"{len(self.rater_categories)} rater categories, "
        ans += f"{self.num_raters_per_category} raters/category, "
        ans += f"{len(self.system_categories)} system categories, "
        ans += f"{self.num_systems_per_category} systems/category)"
        return ans

    def __repr__(self):
        """Return the official string representation of Dataset."""
        return str(self)

    def truncate(self, scores):
        """
        Truncate given scores to range [``min_score``, ``max_score``].

        Parameters
        ----------
        scores : numpy.ndarray
            Input array of scores to be truncated.

        Returns
        -------
        truncated_scores : numpy.ndarray
            Output array with each element of ``scores`` truncated to
            the range [``min_score``, ``max_score``].
        """
        truncated_scores = np.where(scores > self.max_score,
                                    self.max_score,
                                    np.where(scores < self.min_score,
                                             self.min_score,
                                             scores)
                                    )
        return truncated_scores

    def _add_noise_to_scores(self,
                             scores,
                             error_sd,
                             seed,
                             round=True,
                             truncate=True):
        """
        Add noise/error to the given scores.

        The noise/error terms are computed by sampling from a normal distribution
        with a mean of 0 and the given error std. dev.

        This method is useful for generating scores assigned by a hypothetical
        human or automated rater that are usually defined as true scores + error
        in measurement theory.

        Parameters
        ----------
        scores : numpy.ndarray
            The scores to which noise is to be added
        error_sd : float
            The std. dev. of the error term.
        seed : int
            The seed used to instantiate the ``numpy.random.RandomState``
            instance from which the error terms are sampled.
        round : bool, optional
            Whether to round the computed rater scores.
            Defaults to ``True``.
        truncate : bool, optional
            Whether to truncate the computed rater scores to the score
            range defined for this dataset.
            Note that truncation happens after rounding unless ``round``
            is ``False``.
            Defaults to ``True``.

        Returns
        -------
        rater_scores : numpy.ndarray
            Array of scores with noise added to original scores.
        """
        # instantiate a PRNG with the given seed
        prng = np.random.RandomState(seed)

        # sample the error terms from the appropriate distribution
        sampled_errors = prng.normal(0, error_sd, self.num_responses)

        # add the error terms to the true scores and round
        computed_scores = scores + sampled_errors

        # if requested, round the scores to integers first
        rounded_scores = np.round(computed_scores) if round else computed_scores

        # if requested, truncate scores to the range defined for this dataset
        truncated_scores = self.truncate(rounded_scores) if truncate else rounded_scores

        return truncated_scores

    def _generate_true_scores(self, seed):
        """
        Generate true scores for simulated dataset.

        This method simulates true scores based on the pre-defined
        mean and standard deviation. The scores are sampled from the
        normal distribution defined by ``train_score_mean`` and
        ``train_score_sd``. and are truncted to be in the range
        [``min_score``, ``max_score``].

        The generated scores are saved in the private ``_true_scores``
        attribute.

        Parameters
        ----------
        seed : int
            The seed used to instantiate the ``numpy.random.RandomState``
            instance that generates the simulated true scores.
        """
        # instantiate a PRNG with the given seed
        prng = np.random.RandomState(seed)

        # generate true/gold standard score from the normal distribution
        # defined by the mean and standard deviation in settings
        sampled_scores = prng.normal(self.true_score_mean,
                                     self.true_score_sd,
                                     self.num_responses)

        # truncate the scores to the desired range
        true_scores = self.truncate(sampled_scores)
        self._true_scores = true_scores

    def _find_best_error_sd_for_rho(self, rho, error_seed):
        """
        Do a linear search for the best error std. dev. value.

        Search for the error std. dev. value that gets us closest
        to the given rho as follows:
        (a) define a range of error std. dev. values
        (b) generate error terms with each error std. dev. and compute
            hypothetical rater scores by adding noise to the true scores
        (c) compute the average inter-rater correlation for these scores
        (d) return the error std. dev. value for which this average
            inter-rater correlation is closest to our desired rho.

        Parameters
        ----------
        rho : float
            The desired inter-rater correlation.
        error_seed : int
            The seed used to instantiate the ``numpy.random.RandomState``
            instance which is then used to define the normal distribution
            from which error terms are sampled.
        Returns
        -------
        error_sds : numpy.ndarray
            The array of error std. dev. values that is searched.
        mean_correlations : numpy.ndarray
            The array of mean inter-rater correlations corresponding
            to each error std. dev. value
        chosen_error_sd : float
            The chosen error std. dev. value that yields the mean
            inter-rater correlation closest to our desired rho.
        """
        # set up an array of error std. dev. we will search over
        error_sds = np.arange(0.01, 1.5, step=0.01)

        # instantiate an empty list that will hold all the average inter-rater
        # correlation for each error std. dev. value
        mean_inter_rater_correlations = []

        # sweep over the error std. dev. values
        for error_sd in error_sds:

            # instantiate a list that will hold the scores for all human raters
            scores_for_all_raters = []

            # for each rater in this category, compute its hypothetical scores
            # and save them in the list we instantiated above
            for num_rater in range(self.num_raters_per_category):
                rater_seed = num_rater * 25
                scores_for_this_rater = self._add_noise_to_scores(self._true_scores,
                                                                  error_sd,
                                                                  error_seed + rater_seed)
                scores_for_all_raters.append(pd.Series(scores_for_this_rater))

            # convert the 50 x 10000 matrix into a dataframe for convenience
            df_all_rater_scores = pd.concat(scores_for_all_raters, axis=1)

            # compute the correlations between all of the rater scores
            inter_rater_correlations_for_error = df_all_rater_scores.corr().values

            # discard each rater's perfect self-correlation and compute the
            # mean of the remaining correlations
            mean_inter_rater_correlation_for_error = inter_rater_correlations_for_error[inter_rater_correlations_for_error != 1].mean()

            # save this mean inter-rater correlation corresponding
            # to the current error std. dev. value
            mean_inter_rater_correlations.append(mean_inter_rater_correlation_for_error)

        # find the error std. dev. that yields the inter-rater correlation
        # that is closest to the rho that is desired for this rater category
        # in terms of absolute difference
        mean_inter_rater_correlations = np.array(mean_inter_rater_correlations)
        error_idx = (np.abs(mean_inter_rater_correlations - rho)).argmin()
        chosen_error_sd = error_sds[error_idx]

        return error_sds, mean_inter_rater_correlations, chosen_error_sd

    def _generate_rater_scores_and_metadata(self, seed):
        """
        Generate scores and metadata for each rater in each category.

        These are stored in the ``_rater_scores`` and ``_rater_metadata``
        private attributes respectively.

        Parameters
        ----------
        seed : int
            The seed used to instantiate the ``numpy.random.RandomState``
            instance which is then used to define the normal distribution
            from which error terms are sampled.
        """
        # define different search seeds for each rater category
        search_seeds = [100 * i for i in range(1, len(self.rater_categories) + 1)]

        # iterate over each rater category & its desired rho and search
        # for the error_sd value that gets us closest to the desired rho
        for num_category, (rater_category,
                           rho,
                           error_seed) in enumerate(zip(self.rater_categories,
                                                        self.rater_rho_per_category,
                                                        search_seeds)):
            _, _, chosen_error_sd = self._find_best_error_sd_for_rho(rho, error_seed)

            # take this chosen error std. dev. value and generate scores for
            # each rater in this category with a new seed that is different
            # from the seed we used for searching for the error sd value
            for num_rater in range(self.num_raters_per_category):
                rater_seed = num_rater * 123
                scores_for_this_rater = self._add_noise_to_scores(self._true_scores,
                                                                  chosen_error_sd,
                                                                  seed + rater_seed)
                # save this rater's scores
                self._rater_scores.append(scores_for_this_rater)

                # save this rater's metadata
                # which rater number is this overall, not just within the category?
                num_rater_overall = num_rater + num_category * self.num_raters_per_category
                self._rater_metadata.append({"rater_id": f"h_{num_rater_overall + 1}",
                                             "error_sd": chosen_error_sd,
                                             "rater_category": rater_category,
                                             "expected_rho": rho})

    def _generate_system_scores_and_metadata(self, seed):
        """
        Generate scores and metadata for each system in each category.

        These are stored in the ``_system_scores`` and ``_system_metadata``
        private attributes respectively.

        Parameters
        ----------
        seed : int
            The seed used to instantiate the ``numpy.random.RandomState``
            instance which is then used to define the normal distribution
            from which error terms are sampled.
        """
        # iterate over each system category and its desired r2
        for num_category, (system_category,
                           r2) in enumerate(zip(self.system_categories,
                                                self.system_r2_per_category)):

            # note that when it comes to systems, we do not need to _search_
            # for an error std. dev. value that will give us the desired
            # R^2, but we can just analytically solve for it by taking
            # advantage of the following equation:
            # R^2 = 1 - Var(MSE)/Var(True)
            solved_error_sd = np.sqrt(np.var(self._true_scores) * (1 - r2))

            # take this solved error std. dev. value and generate scores for
            # each system in this category with a new seed
            for num_system in range(self.num_systems_per_category):
                system_seed = num_system * 456
                scores_for_this_system = self._add_noise_to_scores(self._true_scores,
                                                                   solved_error_sd,
                                                                   seed + system_seed,
                                                                   round=False)
                # save this system's scores
                self._system_scores.append(scores_for_this_system)

                # save this system's metadata
                # which system number is this overall, not just within the category?
                num_system_overall = num_system + num_category * self.num_systems_per_category
                self._system_metadata.append({"system_id": f"sys_{num_system_overall + 1}",
                                              "system_category": system_category,
                                              "expected_r2_true": r2})

    def fit(self):
        """
        Generate the simulated true, rater, and system scores.

        This method generates the simulated true scores, generates the
        simulated rater scores (and metadata), and generates the simulate
        system scores (and metadata).
        """
        # first generate the true scores
        sys.stderr.write('generating true scores ...\n')
        self._generate_true_scores(12345)

        # generate the rater scores and metadata if they don't already
        # exist or if ``force`` is specified
        sys.stderr.write('generating rater scores and metadata ...\n')
        self._generate_rater_scores_and_metadata(34567)

        # generate the system scores and metadata if they don't already
        # exist or if ``force`` is specified
        sys.stderr.write('generating system scores and metadata ...\n')
        self._generate_system_scores_and_metadata(67890)

    def to_frames(self):
        """
        Return data frames representing this dataset.

        This method generates three data frames containing the simulated
        scores and metadata in this dataset. Note that this method should
        only be called after the dataset has been ``fit()`` and all the
        underlying simulated scores have been generated.

        Returns
        -------
        df_scores : pandas.DataFrame
            The data frame containing the acutal simulated scores for each
            of the ``num_responses`` hypothetical responses in the dataset.
            It has the following columns:
            1. "response_id" : this column contains the id for each
                hypothetical responses
            2. "true" : this column contains the simulated true score
                for each responses.
            3. "h_X" : this column contains the score for each response
                assigned by the simulated rater with ID `h_X`. There are
                ``num_raters_per_category`` * len(rater_categories)``
                such columns.
            3. "sys_X" : this column contains the score for each response
                assigned by the simulated system with ID `sys_X`. There are
                ``num_systems_per_category`` * len(system_categories)``
                such columns.
        df_rater_metadata : pandas.DataFrame
            The data frame containing the metadata for the simulated human
            raters in the dataset. Each row corresponds to one of the
            simulated human raters. It has the following columns:
            1. "rater_id" : this column contains an ID for the simulated
                rater. It is of the form `h_X`, where X goes from 1 to
                ``num_raters_per_category`` * len(rater_categories)``.
            2. "error_sd" : this column contains the value of the error
                std. dev. that was used to simulate the scores for each
                rater. This value was chosen as to get a mean inter-rater
                correlation within this rater's category to be as close
                as possible to the desired rho value for the category.
            3. "rater_category" : this column contains the category
                label that this simulated human rater belongs to.
            4. "expected_rho" : this column contains the desired rho
                for the rater category that this simulated rater
                belongs to.
        df_system_metadata : pandas.DataFrame:
            The data frame containing the metadata for the simulated human
            raters in the dataset. Each row corresponds to one of the
            simulated human raters. It has the following columns:
            1. "system_id" : this column contains an ID for the simulated
                system. It is of the form `sys_X`, where X goes from 1 to
                ``num_systems_per_category`` * len(system_categories)``.
            2. "system_category" : this column contains the category
                label that this simulated system belongs to.
            3. "expected_r2_true" : this column contains the desired R^2
                for the system category that this simulated system
                belongs to.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not already been called.
        """
        if (self._true_scores is None or
                len(self._rater_scores) == 0 or
                len(self._system_scores) == 0):

            raise RuntimeError("This method must be called after the dataset "
                               "has been fit. Call ``self.fit()`` before calling "
                               "this method.")
        else:

            # initialize a dictionary that will hold the various scores
            data_dict = {}

            # create some IDs for the hypothetical responses in the dataset
            data_dict['response_id'] = [f"id_{num_response + 1}" for num_response
                                        in range(self.num_responses)]

            # save the true scores
            data_dict['true'] = self._true_scores

            # save the rater scores
            for rater_scores, rater_metadata in zip(self._rater_scores,
                                                    self._rater_metadata):
                data_dict[rater_metadata['rater_id']] = rater_scores

            # save the system scores
            for system_scores, _system_metadata in zip(self._system_scores,
                                                       self._system_metadata):
                data_dict[_system_metadata['system_id']] = system_scores

            # create the data frames we will return
            df_scores = pd.DataFrame(data_dict)
            df_rater_metadata = pd.DataFrame.from_records(self._rater_metadata)
            df_system_metadata = pd.DataFrame.from_records(self._system_metadata)

            return df_scores, df_rater_metadata, df_system_metadata
