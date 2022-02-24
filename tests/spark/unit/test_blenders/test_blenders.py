import pickle
import random
from typing import cast

import pandas as pd
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession

from lightautoml.automl.blend import BestModelSelector, Blender, WeightedBlender
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.spark.automl.blend import SparkBestModelSelector, SparkWeightedBlender

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM as SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.ml.nested_ml_pipe import SparkNestedTabularMLPipeline as SparkNestedTabularMLPipeline
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.tasks import Task
from .. import from_pandas_to_spark, spark as spark_sess, compare_obtained_datasets

import numpy as np
import pyspark.sql.functions as F

from ..test_auto_ml.utils import DummySparkMLPipeline

spark = spark_sess

def do_compare_blenders(spark: SparkSession, lama_blender: Blender, spark_blender: Blender, to_vector: bool = False):
    with open("unit/resources/datasets/dump_tabular_automl_lgb_linear/Lpred_0_before_blender_before_blender.pickle", "rb") as f:
        data_1, target_1, features_1, roles_1 = pickle.load(f)
        target_1 = pd.Series(target_1)
        nds_1 = NumpyDataset(data_1, features_1, roles_1, task=Task("binary"), target=target_1)

    # with open("test_ml_algo/datasets/dump_tabular_automl_lgb_linear/Lpred_1_before_blender_before_blender.pickle", "rb") as f:
    #     data_2, target_2, features_2, roles_2 = pickle.load(f)
    #     nds_2 = NumpyDataset(data_2, features_2, roles_2, task=Task("binary"))
    #
    # level_preds = [nds_1, nds_2]
    lama_level_preds = [nds_1]
    spark_level_preds = [from_pandas_to_spark(
        nds_1.to_pandas(),
        spark,
        target_1,
        task=SparkTask(name="binary"),
        to_vector=to_vector
    )]

    linear_l2_model = SparkLinearLBFGS()
    lgbm_model = SparkBoostLGBM()
    # Dummpy pipes
    pipes = [
        SparkNestedTabularMLPipeline(ml_algos=[linear_l2_model]),
        SparkNestedTabularMLPipeline(ml_algos=[lgbm_model])
    ]

    # for the testing purpose, this field doesn't exist
    # until fit_predict is called on the pipe
    pipes[0].ml_algos = [linear_l2_model]
    pipes[1].ml_algos = [lgbm_model]

    lama_ds, _ = lama_blender.fit_predict(lama_level_preds, pipes)
    spark_ds, _ = spark_blender.fit_predict(spark_level_preds, pipes)

    lama_ds = cast(NumpyDataset, lama_ds)
    spark_ds = cast(SparkDataset, spark_ds)

    compare_obtained_datasets(lama_ds, spark_ds)


# def test_best_blender(spark: SparkSession):
#     do_compare_blenders(spark, BestModelSelector(), SparkBestModelSelector())
#
#
# def test_weighted_blender(spark: SparkSession):
#     do_compare_blenders(spark, WeightedBlender(), SparkWeightedBlender())


def test_weighted_blender(spark: SparkSession):
    target_col = "target"
    N = 10
    M = 4

    data = [
        {f"pred_{i}": DenseVector([j for j in range(N)]) for i in range(M)}
        for _ in range(100)
    ]
    for d in data:
        d[target_col] = random.randint(0, 5)
    roles = {
        f"pred_{i}": NumericVectorOrArrayRole(
            size=N,
            element_col_name_template=f"pred_{i}" + "_{}",
            dtype=np.float32,
            force_input=True,
            prob=False
        ) for i in range(M)
    }

    pipes = [DummySparkMLPipeline() for i in range(M)]

    predictions_sdf = spark.createDataFrame(data)
    pred_sds = SparkDataset(data=predictions_sdf, task=SparkTask("multiclass"), roles=roles, target=target_col)

    swb = SparkWeightedBlender()
    blended_sds, filtered_pipes = swb.fit_predict(pred_sds, pipes)
    transformed_preds_sdf = swb.transformer.transform(pred_sds.data)
