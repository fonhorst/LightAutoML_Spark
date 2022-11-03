from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.reader.base import PandasToPandasReader
from sparklightautoml.transformers.categorical import SparkLabelEncoderEstimator, SparkFreqEncoderEstimator, \
    SparkOrdinalEncoderEstimator, SparkCatIntersectionsEstimator, SparkTargetEncoderEstimator, \
    SparkMulticlassTargetEncoderEstimator
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, CatIntersectstions, \
    TargetEncoder, MultiClassTargetEncoder

from sparklightautoml.transformers.scala_wrappers.target_encoder_transformer import TargetEncoderTransformer
from sparklightautoml.utils import SparkDataFrame
from .. import DatasetForTest, compare_sparkml_by_content, spark as spark_sess, compare_sparkml_by_metadata
from ..dataset_utils import get_test_datasets

spark = spark_sess

CV = 5

DATASETS = [

    # DatasetForTest("unit/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("tests/spark/unit/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })


    # DatasetForTest("unit/resources/datasets/house_prices.csv",
    #                columns=["Id", "MSZoning", "WoodDeckSF"],
    #                roles={
    #                    "Id": CategoryRole(np.int32),
    #                    "MSZoning": CategoryRole(str),
    #                    "WoodDeckSF": CategoryRole(bool)
    #                })
]


# noinspection PyShadowingNames
@pytest.mark.parametrize("dataset", DATASETS)
def test_sparkml_label_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkLabelEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_metadata(spark, ds, LabelEncoder(), transformer, compare_feature_distributions=True)


# noinspection PyShadowingNames
@pytest.mark.parametrize("dataset", DATASETS)
def test_freq_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkFreqEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_content(spark, ds, FreqEncoder(), transformer)


# noinspection PyShadowingNames
@pytest.mark.parametrize("dataset", DATASETS)
def test_ordinal_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkOrdinalEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_content(spark, ds, OrdinalEncoder(), transformer)


# noinspection PyShadowingNames
@pytest.mark.parametrize("dataset", DATASETS)
def test_cat_intersections(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    # read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    # pdf = pd.read_csv(config['path'], **read_csv_args)
    #
    # reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    # train_ds = reader.fit_read(pdf, roles=config['roles'])
    #
    # # ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    # le_cols = get_columns_by_role(train_ds, "Category")
    # train_ds = train_ds[:, le_cols]
    #
    transformer = SparkCatIntersectionsEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    #
    compare_sparkml_by_metadata(spark, ds, CatIntersectstions(), transformer, compare_feature_distributions=True)
    # compare_sparkml_by_content(spark, ds, CatIntersectstions(), transformer)


def test_scala_target_encoder_transformer(spark: SparkSession):
    enc = {
        "a": [0.0, -1.0, -2.0, -3.0, -4.0],
        "b": [0.0, -1.0, -2.0],
        "c": [0.0, -1.0, -2.0, -3.0]
    }

    oof_enc = {
        "a": [
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [0.0, 11.0, 12.0, 13.0, 14.0],
            [0.0, 21.0, 22.0, 23.0, 24.0]
        ],
        "b": [
            [0.0, 10.0, 20.0],
            [0.0, 11.0, 12.0],
            [0.0, 21.0, 22.0]
        ],
        "c": [
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 11.0, 12.0, 13.0],
            [0.0, 21.0, 22.0, 23.0]
        ]
    }

    fold_column = "fold"
    input_cols = ["a", "b", "c"]
    output_cols = [f"te_{col}" for col in input_cols]
    in_cols = ["id", fold_column, "some_other_col", *input_cols]
    out_cols = [*in_cols, *output_cols]

    def make_df(data: List[List[float]]) -> SparkDataFrame:
        df_data = [
            {col: val for col, val in zip(in_cols, row)}
            for row in data
        ]
        return spark.createDataFrame(df_data)

    data = [
        [0, 0, 42, 1, 1, 1],
        [1, 0, 43, 2, 1, 3],
        [2, 1, 44, 1, 2, 3],
        [3, 1, 45, 1, 2, 2],
        [4, 2, 46, 3, 1, 1],
        [5, 2, 47, 4, 1, 2],
    ]

    result_enc = [
        [0, 0, 42, 1, 1, 1, -1.0, -1.0, -1.0],
        [1, 0, 43, 2, 1, 3, -2.0, -1.0, -3.0],
        [2, 1, 44, 1, 2, 3, -1.0, -2.0, -3.0],
        [3, 1, 45, 1, 2, 2, -1.0, -2.0, -2.0],
        [4, 2, 46, 3, 1, 1, -3.0, -1.0, -1.0],
        [5, 2, 47, 4, 1, 2, -4.0, -1.0, -2.0],
    ]

    result_oof_enc = [
        [0, 0, 42, 1, 1, 1, 10.0, 10.0, 10.0],
        [1, 0, 43, 2, 1, 3, 20.0, 10.0, 30.0],
        [2, 1, 44, 1, 2, 3, 11.0, 12.0, 13.0],
        [3, 1, 45, 1, 2, 2, 11.0, 12.0, 12.0],
        [4, 2, 46, 3, 1, 1, 23.0, 21.0, 21.0],
        [5, 2, 47, 4, 1, 2, 24.0, 21.0, 22.0],
    ]

    data_df = make_df(data)
    target_enc_df = make_df(result_enc)
    target_oof_enc_df = make_df(result_oof_enc)

    tet = TargetEncoderTransformer(enc=enc, oof_enc=oof_enc, fold_column=fold_column, apply_oof=True)\
        .setInputCols(input_cols).setOutputCols(output_cols)

    result_oof_enc_df = tet.transform(data_df)
    result_enc_df = tet.transform(data_df)
    result_enc_df = tet.transform(data_df)


# noinspection PyShadowingNames
@pytest.mark.parametrize("dataset", DATASETS)
def test_target_encoder(spark: SparkSession, dataset: DatasetForTest):
    # reader = PandasToPandasReader(task=Task("binary"), cv=CV, advanced_roles=False)
    # train_ds = reader.fit_read(dataset.dataset, roles=dataset.roles)

    target = pd.Series(np.random.choice(a=[0, 1], size=dataset.dataset.shape[0], p=[0.5, 0.5]))
    folds = pd.Series(np.random.choice(a=[i for i in range(CV)],
                                       size=dataset.dataset.shape[0], p=[1.0 / CV for _ in range(CV)]))

    train_ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"), target=target, folds=folds)

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(spark, train_ds, TargetEncoder(), transformer, compare_feature_distributions=True)


# noinspection PyShadowingNames
@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(dataset="used_cars_dataset")])
def test_target_encoder_real_datasets(spark: SparkSession, config: Dict[str, Any], cv: int):
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    pdf = pd.read_csv(config['path'], **read_csv_args)

    reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    train_ds = reader.fit_read(pdf, roles=config['roles'])

    le_cols = get_columns_by_role(train_ds, "Category")
    train_ds = train_ds[:, le_cols]

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(spark, train_ds, TargetEncoder(), transformer, compare_feature_distributions=True)


# noinspection PyShadowingNames
@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(dataset='ipums_97')])
def test_multi_target_encoder(spark: SparkSession, config: Dict[str, Any], cv: int):
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    pdf = pd.read_csv(config['path'], **read_csv_args)

    reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    train_ds = reader.fit_read(pdf, roles=config['roles'])

    le_cols = get_columns_by_role(train_ds, "Category")
    train_ds = train_ds[:, le_cols]

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkMulticlassTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(
        spark,
        train_ds,
        MultiClassTargetEncoder(),
        transformer,
        compare_feature_distributions=True
    )
