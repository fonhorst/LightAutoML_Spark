import pickle
from typing import cast

from pyspark.sql import SparkSession

from lightautoml.automl.blend import BestModelSelector
from lightautoml.spark.pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline as SparkNestedTabularMLPipeline
from lightautoml.spark.automl.blend import BestModelSelector as SparkBestModelSelector
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS as SparkLinearLBFGS
from lightautoml.tasks import Task
from . import spark


def test_best_blender(spark: SparkSession):
    with open("test_ml_algo/datasets/Lpred_0_before_blender_before_blender.pickle", "rb") as f:
        data_1, target_1, features_1, roles_1 = pickle.load(f)

    with open("test_ml_algo/datasets/Lpred_1_before_blender_before_blender.pickle", "rb") as f:
        data_2, target_2, features_2, roles_2 = pickle.load(f)

    nds_1 = NumpyDataset(data_1, features_1, roles_1, task=Task("binary"))
    nds_2 = NumpyDataset(data_2, features_2, roles_2, task=Task("binary"))

    level_preds = [nds_1, nds_2]

    linear_l2_model = SparkLinearLBFGS()
    # Dummpy pipes
    pipes = [
        SparkNestedTabularMLPipeline(ml_algos=[linear_l2_model]),
        SparkNestedTabularMLPipeline(ml_algos=[None])
    ]

    general_params = {
        "use_algos": ["lgb", "linear_l2"]
    }

    lama_blender = BestModelSelector()
    spark_blender = SparkBestModelSelector()

    lama_blender.fit_predict(level_preds, pipes)
    spark_blender.fit_predict(level_preds, pipes)
