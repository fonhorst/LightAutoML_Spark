import logging.config
import logging.config

from pyspark.sql import SparkSession

from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session, log_exec_time
from lightautoml.spark.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    seed = 100
    path = "/opt/spark_data/sampled_app_train.csv"
    task_type = "binary"
    roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}
    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }
    cacher_key = "main_cache"

    with log_exec_time():

        df = spark.read.csv(path, header=True, escape="\"")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
        sdataset = sreader.fit_read(df, roles=roles)

        iterator = SparkFoldsIterator(sdataset, n_folds=3)

        spark_ml_algo = SparkBoostLGBM(cacher_key=cacher_key, freeze_defaults=False)
        spark_features_pipeline = SparkLGBAdvancedPipeline(cacher_key=cacher_key, **ml_alg_kwargs)
        spark_selector = ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(cacher_key='preselector'),
            ml_algo=SparkBoostLGBM(cacher_key=cacher_key, freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        )

        ml_pipe = SparkMLPipeline(
            cacher_key=cacher_key,
            ml_algos=[spark_ml_algo],
            pre_selection=spark_selector,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        oof_preds_ds = ml_pipe.fit_predict(iterator)
        oof_score = score(oof_preds_ds)
        logger.info(f"OOF score: {oof_score}")

        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_preds_ds = ml_pipe.predict(test_sds)
        test_score = score(test_preds_ds)
        logger.info(f"Test score: {oof_score}")

    logger.info("Finished")

    spark.stop()
