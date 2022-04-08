#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpPermutationImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.automl.base import SparkAutoML
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.pipelines.selection.permutation_importance_based import SparkNpPermutationImportanceEstimator
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.tasks import Task

from examples_utils import get_spark_session
from pyspark.sql import functions as F


if __name__ == "__main__":
    spark = get_spark_session()

    np.random.seed(42)

    print("Load data...")
    data = pd.read_csv("examples/data/sampled_app_train.csv")
    print("Data loaded")

    print("Features modification from user side...")
    data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(str)
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)

    data["constant"] = 1
    data["allnan"] = np.nan

    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    print("Features modification finished")

    print("Split data...")
    train_data, test_data = train_test_split(data, test_size=2000, stratify=data["TARGET"], random_state=13)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    print("Data splitted. Parts sizes: train_data = {}, test_data = {}".format(train_data.shape, test_data.shape))


    train_data_sdf = spark.createDataFrame(train_data).cache()
    test_data_sdf = spark.createDataFrame(test_data).cache()
    train_data_sdf.write.mode('overwrite').format('noop').save()
    test_data_sdf.write.mode('overwrite').format('noop').save()
    

    print("Create task..")
    task = SparkTask("binary")
    print("Task created")

    cacher_key = "main_cache"

    print("Create reader...")
    # reader = PandasToPandasReader(task, cv=5, random_state=1)
    sreader = SparkToSparkReader(task=task, cv=5, random_state=1, advanced_roles=False)
    print("Reader created")

    # selector parts
    print("Create feature selector")
    model01 = SparkBoostLGBM(
        cacher_key='preselector',
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
            # "seed": 42,
            "numThreads": 5,
        }
    )
    model02 = SparkBoostLGBM(
        cacher_key='preselector',
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
            # "seed": 42,
            "numThreads": 5,
        }
    )
    pipe0 = SparkLGBSimpleFeatures(cacher_key='preselector')
    pie = SparkNpPermutationImportanceEstimator()
    pie1 = ModelBasedImportanceEstimator()
    sel1 = ImportanceCutoffSelector(pipe0, model01, pie1, cutoff=0)
    sel2 = NpIterativeFeatureSelector(pipe0, model02, pie, feature_group_size=1, max_features_cnt_in_result=15)
    selector = ComposedSelector([sel1, sel2])
    print("Feature selector created")

    # pipeline 1 level parts
    print("Start creation pipeline_1...")
    pipe = SparkLGBSimpleFeatures(cacher_key=cacher_key)

    print("\t ParamsTuner1 and Model1...")
    params_tuner1 = OptunaTuner(n_trials=100, timeout=100)
    model1 = SparkBoostLGBM(
        cacher_key=cacher_key,
        default_params={
            "learningRate": 0.05,
            "numLeaves": 128,
            # "seed": 1,
            "numThreads": 5,
        }
    )
    print("\t Tuner1 and model1 created")

    print("\t ParamsTuner2 and Model2...")
    model2 = SparkBoostLGBM(
        cacher_key=cacher_key,
        default_params={
            "learningRate": 0.025,
            "numLeaves": 64,
            # "seed": 2,
            "numThreads": 5,
        }
    )
    print("\t Tuner2 and model2 created")

    print("\t Pipeline1...")
    pipeline_lvl1 = SparkMLPipeline(
        cacher_key=cacher_key,
        ml_algos=[(model1, params_tuner1), model2],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )
    print("Pipeline1 created")

    # pipeline 2 level parts
    print("Start creation pipeline_2...")
    pipe1 = SparkLGBSimpleFeatures(cacher_key=cacher_key)

    print("\t ParamsTuner and Model...")
    model = SparkBoostLGBM(
        cacher_key=cacher_key,
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
            "maxBin": 1024,
            # "seed": 3,
            "numThreads": 5,
        }
    )
    print("\t Tuner and model created")

    print("\t Pipeline2...")
    pipeline_lvl2 = SparkMLPipeline(
        cacher_key=cacher_key,
        ml_algos=[model],
        pre_selection=None,
        features_pipeline=pipe1,
        post_selection=None
    )
    print("Pipeline2 created")

    print("Create AutoML pipeline...")
    automl = SparkAutoML(
        sreader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False
    )

    print("AutoML pipeline created...")

    print("Start AutoML pipeline fit_predict...")
    start_time = time.time()
    oof_pred = automl.fit_predict(train_data_sdf, roles={"target": "TARGET"})
    print("AutoML pipeline fitted and predicted. Time = {:.3f} sec".format(time.time() - start_time))

    print("Feature importances of selector:\n{}".format(selector.get_features_score()))

    print("oof_pred:\n{}\nShape = {}".format(oof_pred, oof_pred.shape))

    print("Feature importances of top level algorithm:\n{}".format(automl.levels[-1][0].ml_algos[0].get_features_score()))

    print(
        "Feature importances of lowest level algorithm - model 0:\n{}".format(
            automl.levels[0][0].ml_algos[0].get_features_score()
        )
    )

    print(
        "Feature importances of lowest level algorithm - model 1:\n{}".format(
            automl.levels[0][0].ml_algos[1].get_features_score()
        )
    )

    test_pred = automl.predict(test_data_sdf)
    print("Prediction for test data:\n{}\nShape = {}".format(test_pred, test_pred.shape))

    print("Check scores...")
    score = task.get_dataset_metric()
    off_score = score(oof_pred)
    test_score = score(test_pred)
    print(f"OOF score: {off_score}")
    print(f"TEST score: {test_score}")

    spark.stop()
