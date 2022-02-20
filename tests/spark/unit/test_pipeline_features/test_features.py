from typing import Dict, Any, cast

import numpy as np
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
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


def compare_feature_pipelines(spark: SparkSession, cv: int, ds_config: Dict[str, Any],
                              lama_clazz, slama_clazz, ml_alg_kwargs: Dict[str, Any]):
    spark_ds = prepared_datasets(spark, cv, [ds_config], checkpoint_dir='/opt/test_checkpoints/')
    spark_ds = spark_ds[0]

    # LAMA pipeline
    read_csv_args = {'dtype':  ds_config['dtype']} if 'dtype' in ds_config else dict()
    pdf = pd.read_csv(ds_config['path'], **read_csv_args)
    reader = PandasToPandasReader(task=Task(spark_ds.task.name), cv=cv, advanced_roles=False)
    ds = reader.fit_read(pdf, roles=ds_config['roles'])

    lama_pipeline = lama_clazz(**ml_alg_kwargs)
    lama_feats = lama_pipeline.fit_transform(ds)
    lf_pds = cast(PandasDataset, lama_feats.to_pandas())

    # SLAMA pipeline
    slama_pipeline = slama_clazz(**ml_alg_kwargs)
    slama_pipeline.input_roles = spark_ds.roles
    slama_feats = slama_pipeline.fit_transform(spark_ds)
    slama_lf_pds = cast(PandasDataset, slama_feats.to_pandas())

    # assert sorted(slama_pipeline.output_features) == sorted(lf_pds.features)
    assert sorted(slama_pipeline.output_features) == sorted([f for f in lf_pds.features if not f.startswith('nanflg_')])
    assert len(set(spark_ds.features).difference(slama_feats.features)) == 0
    assert len(set(ds.features).difference(slama_feats.features)) == 0
    assert set(slama_pipeline.output_roles.keys()) == set(f for f in lf_pds.roles.keys() if not f.startswith('nanflg_'))
    assert all([(f in slama_feats.roles) for f in lf_pds.roles.keys() if not f.startswith('nanflg_')])

    not_equal_roles = [
        feat
        for feat, prole in lf_pds.roles.items()
        if not feat.startswith('nanflg_') and
           not (type(prole) == type(slama_pipeline.output_roles[feat]) == type(slama_feats.roles[feat]))
    ]
    assert len(not_equal_roles) == 0, f"Roles are different: {not_equal_roles}"


@pytest.mark.parametrize("ds_config,cv", [(ds, 3) for ds in get_test_datasets(setting="multiclass")])
def test_linear_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    compare_feature_pipelines(spark, cv, ds_config, LinearFeatures, SparkLinearFeatures, ml_alg_kwargs)


@pytest.mark.parametrize("ds_config,cv", [(ds, 3) for ds in get_test_datasets(setting="all-tasks")])
def test_lgbadv_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    compare_feature_pipelines(spark, cv, ds_config, LGBAdvancedPipeline, SparkLGBAdvancedPipeline, ml_alg_kwargs)


@pytest.mark.parametrize("ds_config,cv", [(ds, 3) for ds in get_test_datasets(setting="all-tasks")])
def test_lgbsimple_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    ml_alg_kwargs = {}
    compare_feature_pipelines(spark, cv, ds_config, LGBSimpleFeatures, SparkLGBSimpleFeatures, ml_alg_kwargs)
