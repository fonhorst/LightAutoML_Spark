"""Iterative feature selector."""

import logging

from copy import deepcopy
from typing import Optional

import numpy as np

from pandas import Series

from pyspark.sql import functions as F
from pyspark.sql.functions import shuffle

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset, SparkDataset
from ...ml_algo.base import MLAlgo
from lightautoml.spark.pipelines.selection.base import SparkImportanceEstimator


logger = logging.getLogger(__name__)



class NpPermutationImportanceEstimator(SparkImportanceEstimator):
    """Permutation importance based estimator.

    Importance calculate, using random permutation
    of items in single column for each feature.

    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: seed for random generation of features permutation.

        """
        super().__init__()
        self.random_state = random_state

    def fit(
        self,
        train_valid: Optional[TrainValidIterator] = None,
        ml_algo: Optional[MLAlgo] = None,
        preds: Optional[LAMLDataset] = None,
    ):
        """Find importances for each feature in dataset.

        Args:
            train_valid: Initial dataset iterator.
            ml_algo: Algorithm.
            preds: Predicted target values for validation dataset.

        """

        normal_score = ml_algo.score(preds)
        logger.debug("Normal score = {}".format(normal_score))

        valid_data: SparkDataset
        valid_data = train_valid.get_validation_data()
        # valid_data = valid_data.to_numpy()

        # permutation = np.random.RandomState(seed=self.random_state).permutation(valid_data.shape[0])
        permutation_importance = {}

        for it, col in enumerate(valid_data.features):
            logger.debug("Start processing ({},{})".format(it, col))
            # Save initial column
            # save_col = deepcopy(valid_data[:, col])

            # Get current column and shuffle it
            # shuffled_col = valid_data[permutation, col]

            # Set shuffled column
            # logger.info3("Shuffled column set")
            # valid_data[col] = shuffled_col


            df = valid_data.data
            ds: SparkDataset = valid_data.empty()
            ds.set_data(
                df.select(
                    *[c for c in valid_data.data.columns if c != col],
                    (shuffle(F.col(col))).alias(col)
                ),
                valid_data.features,
                valid_data.roles,
                valid_data.dependencies
            )

            # Calculate predict and metric
            logger.info3("Shuffled column set")
            new_preds = ml_algo.predict(ds)
            shuffled_score = ml_algo.score(new_preds)
            logger.debug(
                "Shuffled score for col {} = {}, difference with normal = {}".format(
                    col, shuffled_score, normal_score - shuffled_score
                )
            )
            permutation_importance[col] = normal_score - shuffled_score

            # Set normal column back to the dataset
            # logger.debug("Normal column set")
            # valid_data[col] = save_col

        self.raw_importances = Series(permutation_importance).sort_values(ascending=False)
