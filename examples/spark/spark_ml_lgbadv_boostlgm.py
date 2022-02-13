from copy import deepcopy
import logging.config

import logging.config
from copy import deepcopy
from typing import cast

import numpy as np
from pyspark.ml import PipelineModel

from lightautoml.dataset.roles import FoldsRole, TargetRole
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.validation.iterators import SparkFoldsIterator
from lightautoml.validation.base import DummyIterator

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
        sreader = SparkToSparkReader(task=SparkTask("reg"), cv=3)
        sdataset_tmp = sreader.fit_read(df, roles=roles)
        
        sdataset = sdataset_tmp.empty()
        new_roles = deepcopy(sdataset_tmp.roles)

        sdataset.set_data(
            sdataset_tmp.data \
                .join(sdataset_tmp.target, SparkDataset.ID_COLUMN) \
                .join(sdataset_tmp.folds, SparkDataset.ID_COLUMN),
            sdataset_tmp.features,
            new_roles
        )

        ml_alg_kwargs = {
            'auto_unique_co': 10,
            'max_intersection_depth': 3,
            'multiclass_te_co': 3,
            'output_categories': True,
            'top_intersections': 4
        }

        # # Spark ML pipeline
        simple_pipline_builder = SparkLGBAdvancedPipeline(sdataset.features, sdataset.roles, **ml_alg_kwargs)
        sdataset_feats = simple_pipline_builder.fit_transform(sdataset)

        iterator = SparkFoldsIterator(sdataset_feats, n_folds=3)
        spark_ml_algo = BoostLGBM(input_cols=simple_pipline_builder.output_features, freeze_defaults=False)
        spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)
        spark_ml_algo = cast(BoostLGBM, spark_ml_algo)

        final = PipelineModel(stages=[simple_pipline_builder.transformer, spark_ml_algo.transformer])

        final_result = final.transform(sdataset_tmp.data)
        final_result.write.mode('overwrite').format('noop').save()

        logger.info("Finished")
