#!/usr/bin/env python
# coding: utf-8

"""
building ML pipeline from blocks and fit + predict the pipeline itself
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import FoldsRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.dataset.utils import roles_parser
from lightautoml.ml_algo.tuning.optuna import OptunaTuner


from lightautoml.pipelines.selection.importance_based import (
    ImportanceCutoffSelector,
    ModelBasedImportanceEstimator,
)
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.validation.iterators import SparkFoldsIterator

from examples_utils import get_spark_session
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel


if __name__ == "__main__":
    spark = get_spark_session()

    # Read data from file
    print("Read data from file")
    data = pd.read_csv(
        "examples/data/sampled_app_train.csv",
        usecols=[
            "TARGET",
            "NAME_CONTRACT_TYPE",
            "AMT_CREDIT",
            "NAME_TYPE_SUITE",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ],
    )

    # Fix dates and convert to date type
    print("Fix dates and convert to date type")
    data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(str)
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    # Create folds
    print("Create folds")
    data["__fold__"] = np.random.randint(0, 5, len(data))

    # Print data head
    print("Print data head")
    print(data.head())

    dataset_sdf = spark.createDataFrame(data)
    dataset_sdf = dataset_sdf.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN)
    ).cache()
    dataset_sdf.write.mode('overwrite').format('noop').save()
    dataset_sdf = dataset_sdf.select(F.col("__fold__").cast("int").alias("__fold__"), *[c for c in dataset_sdf.columns if c != "__fold__"])
    

    # # Set roles for columns
    print("Set roles for columns")
    check_roles = {
        TargetRole(): "TARGET",
        CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
        NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
        DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
        FoldsRole(): "__fold__",
    }

    # create Task
    task = SparkTask("binary")
    cacher_key = "main_cache"

    # # Creating PandasDataSet
    print("Creating PandasDataset")
    start_time = time.time()
    # pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
    sreader = SparkToSparkReader(task=task, advanced_roles=False)
    sdataset = sreader.fit_read(dataset_sdf, roles=check_roles)
    print("PandasDataset created. Time = {:.3f} sec".format(time.time() - start_time))

    # # Print pandas dataset feature roles
    print("Print pandas dataset feature roles")
    roles = sdataset.roles
    for role in roles:
        print("{}: {}".format(role, roles[role]))

    # # Feature selection part
    print("Feature selection part")
    selector_iterator = SparkFoldsIterator(sdataset, 1)
    print("Selection iterator created")

    pipe = SparkLGBSimpleFeatures(cacher_key='preselector')
    print("Pipe and model created")

    model0 = SparkBoostLGBM(
        cacher_key='preselector',
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
            # "seed": 0,
            "numThreads": 5,
        }
    )

    mbie = ModelBasedImportanceEstimator()
    selector = ImportanceCutoffSelector(pipe, model0, mbie, cutoff=10)
    start_time = time.time()
    selector.fit(selector_iterator)
    print("Feature selector fitted. Time = {:.3f} sec".format(time.time() - start_time))

    print("Feature selector scores:")
    print("\n{}".format(selector.get_features_score()))

    # # Build AutoML pipeline
    print("Start building AutoML pipeline")
    pipe = SparkLGBSimpleFeatures(cacher_key=cacher_key)
    print("Pipe created")

    params_tuner1 = OptunaTuner(n_trials=10, timeout=300)
    model1 = SparkBoostLGBM(cacher_key=cacher_key, default_params={"learningRate": 0.05, "numLeaves": 128})
    print("Tuner1 and model1 created")

    params_tuner2 = OptunaTuner(n_trials=100, timeout=300)
    model2 = SparkBoostLGBM(cacher_key=cacher_key, default_params={"learningRate": 0.025, "numLeaves": 64})
    print("Tuner2 and model2 created")

    total = SparkMLPipeline(
        cacher_key=cacher_key,
        ml_algos=[(model1, params_tuner1), (model2, params_tuner2)],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    print("Finished building AutoML pipeline")

    # # Create full train iterator
    print("Full train valid iterator creation")
    train_valid = SparkFoldsIterator(sdataset)
    print("Full train valid iterator created")

    # # Fit predict using pipeline
    print("Start AutoML pipeline fit_predict")
    start_time = time.time()
    pred = total.fit_predict(train_valid)
    print("Fit_predict finished. Time = {:.3f} sec".format(time.time() - start_time))

    # # Check preds
    print("Preds:")
    print("\n{}".format(pred))
    print("Preds.shape = {}".format(pred.shape))

    # # Predict full train dataset
    print("Predict full train dataset")
    start_time = time.time()
    train_pred = total.predict(sdataset)
    print("Predict finished. Time = {:.3f} sec".format(time.time() - start_time))
    print("Preds:")
    print("\n{}".format(train_pred))
    print("Preds.shape = {}".format(train_pred.shape))

    print("Save MLPipeline")
    total.transformer.write().overwrite().save("file:///tmp/SparkMLPipeline")

    print("Load saved MLPipeline")
    pipeline_model = PipelineModel.load("file:///tmp/SparkMLPipeline")

    print("Predict loaded automl")
    preds = pipeline_model.transform(sdataset.data)

    # # # Check preds feature names
    print("Preds columns: {}".format(preds.columns))

    # # Check model feature scores
    print("Feature scores for model_1:\n{}".format(model1.get_features_score()))
    print("Feature scores for model_2:\n{}".format(model2.get_features_score()))

    spark.stop()