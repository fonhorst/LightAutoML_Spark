import logging.config
from typing import cast

from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.automl.blend import SparkWeightedBlender
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.validation.iterators import SparkFoldsIterator

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
        task = SparkTask("reg")
        sreader = SparkToSparkReader(task=task, cv=3)
        sdataset = sreader.fit_read(df, roles=roles)

        ml_alg_kwargs = {
            'auto_unique_co': 10,
            'max_intersection_depth': 3,
            'multiclass_te_co': 3,
            'output_categories': True,
            'top_intersections': 4
        }

        cacher_key = "main_cache"

        iterator = SparkFoldsIterator(sdataset, n_folds=3)

        spark_ml_algo1 = SparkLinearLBFGS(freeze_defaults=False)
        spark_features_pipeline = SparkLinearFeatures(cacher_key=cacher_key, **ml_alg_kwargs)
        spark_selector = ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(cacher_key='preselector'),
            ml_algo=SparkBoostLGBM(freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        )

        spark_ml_algo2 = SparkBoostLGBM(freeze_defaults=False)
        spark_features_pipeline2 = SparkLinearFeatures(cacher_key=cacher_key, **ml_alg_kwargs)
        spark_selector2 = ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(cacher_key='preselector'),
            ml_algo=SparkBoostLGBM(freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        )

        ml_pipe1 = SparkMLPipeline(
            cacher_key=cacher_key,
            input_roles=sdataset.roles,
            ml_algos=[spark_ml_algo1],
            pre_selection=spark_selector,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )
        ml_pipe2 = SparkMLPipeline(
            cacher_key=cacher_key,
            input_roles=sdataset.roles,
            ml_algos=[spark_ml_algo2],
            pre_selection=spark_selector2,
            features_pipeline=spark_features_pipeline2,
            post_selection=None
        )

        predictions = ml_pipe1.fit_predict(iterator)
        sds = cast(SparkDataset, predictions)
        iterator = SparkFoldsIterator(sds, n_folds=3)
        ml_pipe2.input_roles = predictions.roles
        predictions = ml_pipe2.fit_predict(iterator)
        sds = cast(SparkDataset, predictions)
        iterator = SparkFoldsIterator(sds, n_folds=3)

        roles = dict()
        roles.update(ml_pipe1.output_roles)
        roles.update(ml_pipe2.output_roles)
        sds = cast(SparkDataset, iterator.train)
        sdf = sds.data.select(*sds.service_columns, *list(roles.keys()))
        level_predictions = sds.empty()
        level_predictions.set_data(sdf, sdf.columns, roles)

        blender = SparkWeightedBlender()
        blender.fit_predict(level_predictions, [ml_pipe1, ml_pipe2])


        logger.info("Finished")
