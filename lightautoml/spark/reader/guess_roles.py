from typing import Optional, Union

from lightautoml.reader.guess_roles import calc_ginis
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer, SequentialTransformer

from pyspark.sql import functions as F

import pandas as pd
import numpy as np

from lightautoml.spark.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, \
    MultiClassTargetEncoder, TargetEncoder


def get_score_from_pipe(
    train: SparkDataset,
    pipe: Optional[SparkTransformer] = None,
    empty_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train:  np.ndarray.
        target: np.ndarray.
        pipe: LAMLTransformer.
        empty_slice: np.ndarray.

    Returns:
        np.ndarray.

    """

    if pipe is not None:
        train = pipe.fit_transform(train)

    def gini_func(iterator):
        for pdf in iterator:
            target = pdf[train.target_column].to_numpy()
            data = pdf.drop(train.target_column)
            cols = data.columns
            data = data.to_numpy()
            scores = calc_ginis(data, target, empty_slice)
            yield pd.DataFrame(data=scores,
                               columns=cols)
    schema = train.data.schema
    cols = train.data.columns
    mean_scores = (
        train.data
        .join(train.target, on=SparkDataset.ID_COLUMN)
        .mapInPandas(gini_func, schema)
        .select([F.mean(c).alias(c) for c in cols])
    ).toPandas().to_numpy()

    return mean_scores


def get_category_roles_stat(
    train: SparkDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42
):
    """Search for optimal processing of categorical values.

    Categorical means defined by user or object types.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: seed of random numbers generator.
        n_jobs: number of jobs.

    Returns:
        result.

    """

    roles_to_identify = []

    dtypes = []

    # check for train dtypes
    roles = []
    for f in train.features:
        role = train.roles[f]
        if role.name == "Category" and role.encoding_type == "auto":
            roles_to_identify.append(f)
            roles.append(role)
            dtypes.append(role.dtype)

    res = pd.DataFrame(
        columns=[
            "unique",
            "top_freq_values",
            "dtype",
            "encoded_scores",
            "freq_scores",
            "ord_scores",
        ],
        index=roles_to_identify,
    )

    res["dtype"] = dtypes

    if len(roles_to_identify) == 0:
        return res

    train = train.empty()
    sdf = train.data.select(SparkDataset.ID_COLUMN, *roles_to_identify)

    if subsample is not None:
        sdf = sdf.sample(fraction=subsample, seed=random_state)

    train.set_data(sdf, roles_to_identify, roles)

    assert train.folds is not None

    # if train.folds is None:
    #     train.folds = set_sklearn_folds(train.task, train.target.values, cv=5, random_state=42, group=train.group)
    #
    # if subsample is not None:
    #     idx = np.random.RandomState(random_state).permutation(train.shape[0])[:subsample]
    #     train = train[idx]

    # check task specific
    if train.task.name == "multiclass":
        encoder = MultiClassTargetEncoder
    else:
        encoder = TargetEncoder

    # check label encoded scores
    trf = SequentialTransformer([LabelEncoder(), encoder()])
    res["encoded_scores"] = get_score_from_pipe(train, pipe=trf)

    # check frequency encoding
    trf = FreqEncoder()
    res["freq_scores"] = get_score_from_pipe(train, pipe=trf)

    # check ordinal encoding
    trf = OrdinalEncoder()
    res["ord_scores"] = get_score_from_pipe(train, pipe=trf)

    return res