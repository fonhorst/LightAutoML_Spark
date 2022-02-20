from typing import Dict, Any, cast

import numpy as np
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.tasks import Task
from .. import DatasetForTest, spark as spark_sess
from ..dataset_utils import get_test_datasets, prepared_datasets

import pandas as pd

spark = spark_sess

DATASETS = [

    # DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("unit/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })
]


@pytest.mark.parametrize("ds_config,cv", [(ds, 3) for ds in get_test_datasets(setting="fast")])
def test_linear_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    spark_ds = prepared_datasets(spark, cv, [ds_config], checkpoint_dir='/opt/test_checkpoints/')
    spark_ds = spark_ds[0]

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    # LAMA pipeline
    pdf = pd.read_csv(ds_config['path'],  dtype=ds_config['dtype'])
    reader = PandasToPandasReader(task=Task(spark_ds.task.name), cv=cv, advanced_roles=False)
    ds = reader.fit_read(pdf, roles=ds_config['roles'])

    linear_features = LinearFeatures(**ml_alg_kwargs)
    dataset_feats = linear_features.fit_transform(ds)
    lf_pds = cast(PandasDataset, dataset_feats.to_pandas())

    # SLAMA pipeline
    slama_linear_features = SparkLinearFeatures(**ml_alg_kwargs)
    slama_linear_features.input_roles = spark_ds.roles
    slama_dataset_feats = slama_linear_features.fit_transform(spark_ds)
    slama_lf_pds = cast(PandasDataset, slama_dataset_feats.to_pandas())

    assert sorted(slama_linear_features.output_features) == sorted(lf_pds.features)
    assert len(set(spark_ds.features).difference(slama_dataset_feats.features)) == 0
    assert len(set(ds.features).difference(slama_dataset_feats.features)) == 0
    assert slama_linear_features.output_roles == lf_pds.roles
    pass


# @pytest.mark.parametrize("ds_config,cv", [(ds, 3) for ds in get_test_datasets(setting="fast")])
# def test_lgbadv_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
#     spark_ds = prepared_datasets(spark, cv, [ds_config], checkpoint_dir='/opt/test_checkpoints/')
#     spark_ds = spark_ds[0]
#
#     ml_alg_kwargs = {
#         'auto_unique_co': 10,
#         'max_intersection_depth': 3,
#         'multiclass_te_co': 3,
#         'output_categories': True,
#         'top_intersections': 4
#     }
#
#     # LAMA pipeline
#     pdf = pd.read_csv(ds_config['path'],  dtype=ds_config['dtype'])
#     reader = PandasToPandasReader(Task(spark_ds.task.name), cv, advanced_roles=False)
#     ds = reader.fit_read(pdf, roles=ds_config['roles'])
#
#     linear_features = LinearFeatures(**ml_alg_kwargs)
#     dataset_feats = linear_features.fit_transform(ds)
#     lf_pds = cast(PandasDataset, dataset_feats.to_pandas())
#
#     # SLAMA pipeline
#     slama_linear_features = SparkLinearFeatures(**ml_alg_kwargs)
#     slama_linear_features.input_roles = spark_ds.roles
#     slama_dataset_feats = slama_linear_features.fit_transform(spark_ds)
#     slama_lf_pds = cast(PandasDataset, slama_dataset_feats.to_pandas())
#
#     assert sorted(lf_pds.features) == sorted(slama_lf_pds.features)
#     pass

# @pytest.mark.parametrize("dataset", DATASETS)
# def test_linear_features(spark: SparkSession, dataset: DatasetForTest):
#
#     # difference in folds ??
#     ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
#     # sds = SparkDataset.from_lama(ds, spark)
#     sds = from_pandas_to_spark(ds, spark, ds.target)
#
#     # dumped from simple_tabular_classification.py
#     kwargs = {
#         'auto_unique_co': 50,
#         'feats_imp': None,
#         'kwargs': {},
#         'max_bin_count': 10,
#         'max_intersection_depth': 3,
#         'multiclass_te_co': 3,
#         'output_categories': True,
#         'sparse_ohe': 'auto',
#         'subsample': 100000,
#         'top_intersections': 4
#     }
#
#     linear_features = LinearFeatures(**kwargs)
#
#     lama_transformer = linear_features.create_pipeline(ds)
#
#     spark_linear_features = SparkLinearFeatures(**kwargs)
#
#     spark_linear_features.input_roles = sds.roles
#
#     spark_transformer = spark_linear_features.create_pipeline(sds)
#
#     print()
#     print(lama_transformer.print_structure())
#     print()
#     print()
#     print(lama_transformer.print_tr_types())
#
#     print("===================================================")
#
#     print()
#     # print(spark_transformer.print_structure())
#     print()
#     print()
#     # print(spark_transformer.print_tr_types())
#
#     with log_exec_time():
#         lama_ds = linear_features.fit_transform(ds).to_numpy()
#
#     with log_exec_time():
#         spark_ds = spark_linear_features.fit_transform(sds)
#         cutted_spark_ds = spark_ds.empty()
#         sdf = spark_ds.data.select(SparkDataset.ID_COLUMN, *spark_linear_features.output_features)
#         cutted_spark_ds.set_data(sdf,
#                                  spark_linear_features.output_features,
#                                  spark_linear_features.output_roles)
#
#     # time.sleep(600)
#     compare_obtained_datasets(lama_ds, cutted_spark_ds)
#
#
# @pytest.mark.parametrize("dataset", DATASETS)
# def test_lgb_simple_features(spark: SparkSession, dataset: DatasetForTest):
#
#     ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
#     # sds = SparkDataset.from_lama(ds, spark)
#     sds = from_pandas_to_spark(ds, spark, ds.target)
#
#     # no args in simple_tabular_classification.py
#     lgb_features = LGBSimpleFeatures()
#
#     lama_transformer = lgb_features.create_pipeline(ds)
#
#     spark_lgb_features = SparkLGBSimpleFeatures()
#     spark_lgb_features.input_roles = sds.roles
#     spark_transformer = spark_lgb_features.create_pipeline(sds)
#
#     with log_exec_time():
#         lama_ds = lgb_features.fit_transform(ds)
#
#     with log_exec_time():
#         spark_ds = spark_lgb_features.fit_transform(sds)
#         cutted_spark_ds = spark_ds.empty()
#         sdf = spark_ds.data.select(SparkDataset.ID_COLUMN, *spark_lgb_features.output_features)
#         cutted_spark_ds.set_data(sdf,
#                                  spark_lgb_features.output_features,
#                                  spark_lgb_features.output_roles)
#
#     compare_obtained_datasets(lama_ds, cutted_spark_ds)
#
#
# @pytest.mark.parametrize("dataset", DATASETS)
# def test_lgb_advanced_features(spark: SparkSession, dataset: DatasetForTest):
#
#     ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
#     # sds = SparkDataset.from_lama(ds, spark)
#     sds = from_pandas_to_spark(ds, spark, ds.target)
#
#     # dumped from simple_tabular_classification.py
#     kwargs = {
#         'ascending_by_cardinality': False,
#         'auto_unique_co': 10,
#         'feats_imp': None,
#         'max_intersection_depth': 3,
#         'multiclass_te_co': 3,
#         'output_categories': False,
#         'subsample': 100000,
#         'top_intersections': 4
#     }
#
#     lgb_features = LGBAdvancedPipeline(**kwargs)
#     lama_transformer = lgb_features.create_pipeline(ds)
#
#     spark_lgb_features = SparkLGBAdvancedPipeline(**kwargs)
#     spark_lgb_features.input_roles = sds.roles
#     spark_transformer = spark_lgb_features.create_pipeline(sds)
#
#     with log_exec_time():
#         lama_ds = lgb_features.fit_transform(ds)
#
#     with log_exec_time():
#         spark_ds = spark_lgb_features.fit_transform(sds)
#         cutted_spark_ds = spark_ds.empty()
#         sdf = spark_ds.data.select(SparkDataset.ID_COLUMN, *spark_lgb_features.output_features)
#         cutted_spark_ds.set_data(sdf,
#                                  spark_lgb_features.output_features,
#                                  spark_lgb_features.output_roles)
#
#     compare_obtained_datasets(lama_ds, cutted_spark_ds)
