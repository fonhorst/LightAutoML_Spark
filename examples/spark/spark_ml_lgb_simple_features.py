from copy import deepcopy
import logging
import logging.config
import os
from typing import List, cast

from pyspark.ml import Pipeline, PipelineModel

from lightautoml.dataset.roles import FoldsRole, TargetRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBSimpleFeaturesTmp
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask

import numpy as np

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:
        roles = {
            "target": 'price',
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            "numeric": ['latitude', 'longitude', 'mileage']
        }

        # data reading and converting to SparkDataset
        df = spark.read.csv("examples/data/tiny_used_cars_data.csv", header=True, escape="\"")
        sreader = SparkToSparkReader(task=SparkTask("reg"), cv=5)
        sdataset_tmp = sreader.fit_read(df, roles=roles)

        
        sdataset = sdataset_tmp.empty()
        new_roles = deepcopy(sdataset_tmp.roles)
        new_roles.update({sdataset_tmp.target_column: TargetRole(np.float32), sdataset_tmp.folds_column: FoldsRole()})
        sdataset.set_data(
            sdataset_tmp.data \
                .join(sdataset_tmp.target, SparkDataset.ID_COLUMN) \
                .join(sdataset_tmp.folds, SparkDataset.ID_COLUMN),
            sdataset_tmp.features,
            new_roles
        )
        # sdataset.to_pandas().data.to_csv("/tmp/sdataset_data.csv")

        # Spark ML pipeline
        simple_pipline_builder = LGBSimpleFeaturesTmp()
        spark_pipeline = simple_pipline_builder.create_pipeline(sdataset)
        spark_pipeline_model = spark_pipeline.fit(sdataset.data)
        spark_pipeline_model.transform(sdataset.data).toPandas().to_csv("/tmp/spark_pipeline_model.csv", index=False)

        # SparkTransformer pipeline
        lgb_simple_feature_builder = LGBSimpleFeatures()
        pipeline = lgb_simple_feature_builder.create_pipeline(sdataset)
        pipeline.fit_transform(sdataset).to_pandas().data.to_csv("/tmp/pipeline_model.csv", index=False)

        logger.info("Finished")

