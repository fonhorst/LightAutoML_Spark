from typing import Dict, Optional, Union, List

from pyspark.sql.types import IntegerType

from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.guess_roles import calc_ginis, RolesDict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer, SequentialTransformer, ChangeRoles

from pyspark.sql import functions as F, Window

import pandas as pd
import numpy as np

from lightautoml.spark.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, \
    MultiClassTargetEncoder, TargetEncoder
from lightautoml.spark.transformers.numeric import QuantileBinning


def get_gini_func(target_col: str):
    def gini_func(iterator):
        for pdf in iterator:
            target = pdf[target_col].to_numpy()
            data = pdf.drop(target_col, axis=1)
            cols = data.columns
            data = data.to_numpy()
            scores = calc_ginis(data, target, None)
            yield pd.DataFrame(data=[scores],
                               columns=cols)

    return gini_func


def get_score_from_pipe(
    train: SparkDataset,
    pipe: Optional[SparkTransformer] = None
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

    gini_func = get_gini_func(train.target_column)

    sdf = (
        train.data
        .join(train.target, on=SparkDataset.ID_COLUMN)
    )

    mean_scores = (
        sdf
        .mapInPandas(gini_func, train.data.schema)
        .select([F.mean(c).alias(c) for c in train.features])
    ).toPandas().values.flatten()

    return mean_scores


def get_numeric_roles_stat(
    train: SparkDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    manual_roles: Optional[RolesDict] = None) -> pd.DataFrame:
    """Calculate statistics about different encodings performances.

    We need it to calculate rules about advanced roles guessing.
    Only for numeric data.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: int.
        manual_roles: Dict.
        n_jobs: int.

    Returns:
        DataFrame.

    """
    if manual_roles is None:
        manual_roles = {}

    roles_to_identify = []
    roles = []
    flg_manual_set = []
    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == "Numeric":
            roles_to_identify.append(f)
            roles.append(role)
            flg_manual_set.append(f in manual_roles)

    res = pd.DataFrame(
        columns=[
            "flg_manual",
            "unique",
            "unique_rate",
            "top_freq_values",
            "raw_scores",
            "binned_scores",
            "encoded_scores",
            "freq_scores",
            "nan_rate",
        ],
        index=roles_to_identify,
    )
    res["flg_manual"] = flg_manual_set

    if len(roles_to_identify) == 0:
        return res

    sdf = train.data.select(SparkDataset.ID_COLUMN, *roles_to_identify)

    if subsample is not None:
        total_number = sdf.count()
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample/total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
    train.set_data(sdf, roles_to_identify, roles)

    assert train.folds is not None

    # if train.folds is None:
    #     train.folds = set_sklearn_folds(train.task, train.target, cv=5, random_state=42, group=train.group)

    data, target = train.data, train.target

    # check task specific
    if train.task.name == "multiclass":
        encoder = MultiClassTargetEncoder
    else:
        encoder = TargetEncoder

    # s3d = data.shape + (-1,)
    # empty_slice = np.isnan(data)

    # check scores as is
    res["raw_scores"] = get_score_from_pipe(train)

    # check unique values
    sub_select_columns = []
    top_select_columns = []
    for f in train.features:
        sub_select_columns.append(F.count(F.when(~F.isnan(F.col(f)), F.col(f))).over(Window.partitionBy(F.col(f))).alias(f'{f}_count_values'))
        top_select_columns.append(F.max(F.col(f'{f}_count_values')).alias(f'{f}_max_count_values'))
        top_select_columns.append(F.count_distinct(F.when(~F.isnan(F.col(f)), F.col(f))).alias(f'{f}_count_distinct'))
    df = train.data.select(*train.features, *sub_select_columns)
    unique_values_stat: Dict = df.select(*top_select_columns).first().asDict()

    # max of frequency of unique values in every column
    res["top_freq_values"] = np.array([unique_values_stat[f'{f}_max_count_values'] for f in train.features])
    # how many unique values in every column
    res["unique"] = np.array([unique_values_stat[f'{f}_count_distinct'] for f in train.features])
    res["unique_rate"] = res["unique"] / train.shape[0]

    # check binned categorical score
    trf = SequentialTransformer([QuantileBinning(), encoder()])
    res["binned_scores"] = get_score_from_pipe(train, pipe=trf)

    # check label encoded scores
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)), LabelEncoder(), encoder()])
    res["encoded_scores"] = get_score_from_pipe(train, pipe=trf)

    # check frequency encoding
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)), FreqEncoder()])
    res["freq_scores"] = get_score_from_pipe(train, pipe=trf)

    # res["nan_rate"] = empty_slice.mean(axis=0)

    return res


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

    sdf = train.data.select(SparkDataset.ID_COLUMN, *roles_to_identify)

    if subsample is not None:
        total_number = sdf.count()
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample/total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
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


def get_null_scores(
    train: SparkDataset,
    feats: Optional[List[str]] = None,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
) -> pd.Series:
    """Get null scores.

    Args:
        train: Dataset
        feats: list of features.
        subsample: size of subsample.
        random_state: seed of random numbers generator.

    Returns:
        Series.

    """
    roles = train.roles
    sdf = train.data.select(SparkDataset.ID_COLUMN, *feats)

    if subsample is not None:
        total_number = sdf.count()
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample/total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
    train.set_data(sdf, feats, [roles[f] for f in feats])

    train.cache()
    size = train.data.count()
    notnan = train.data.select([
        F.sum(F.isnull(feat).astype(IntegerType())).alias(feat)
        for feat in train.features
    ]).first().asDict()

    notnan_cols = [
        feat for feat, cnt in notnan.items()
        if cnt != size and cnt != 0
    ]

    gini_func = get_gini_func(train.target_column)
    sdf = (
        train.data
        .select(SparkDataset.ID_COLUMN, *notnan_cols)
        .join(train.target, on=SparkDataset.ID_COLUMN)
    )
    if notnan_cols:
        mean_scores = (
            sdf
            .mapInPandas(gini_func, train.data.schema)
            .select([F.mean(c).alias(c) for c in notnan_cols])
        ).first().asDict()
    else:
        mean_scores = {}

    scores = [
        mean_scores[feat] if feat in mean_scores else 0.0
        for feat in train.features
    ]

    train.uncache()

    res = pd.Series(scores, index=train.features, name="max_score")

    return res