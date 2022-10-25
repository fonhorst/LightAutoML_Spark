import logging.config

from pyspark.sql import SparkSession

from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from .utils import DummyTabularAutoML
from .. import spark as spark_sess

spark = spark_sess

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


# noinspection PyShadowingNames
def test_automl_preset(spark: SparkSession):
    n_classes = 10

    train_data = spark.createDataFrame([
        {"a": i, "b": 100 + i, "c": 100 * i, "TARGET": i % n_classes} for i in range(120)
    ])

    test_data = spark.createDataFrame([
        {"a": i, "b": 100 + i, "c": 100 * i, "TARGET": i % n_classes} for i in range(120, 140)
    ])

    persistence_manager = PlainCachePersistenceManager()
    automl = DummyTabularAutoML(n_classes=n_classes)

    # 1. check for output result, features, roles (required columns in data, including return_all_predictions)
    # 2. checking for layer-to-layer data transfer (internal in DummyTabularAutoML):
    #   - all predictions of the first level are available in all pipes of the second level
    #   - all inputs data are presented in all pipes of the first level
    #   - all inputs data are presented in all pipes of the second level (if skip_conn)
    # 3. blending and return_all_predictions works correctly
    oof_ds = automl.fit_predict(
        train_data,
        roles={"target": "TARGET"},
        persistence_manager=persistence_manager
    ).persist()
    pred_ds = automl.predict(test_data, persistence_manager=persistence_manager).persist()

    # ds_names = {ds.uid: ds.name for ds in persistence_manager.all_datasets}
    # print(f"{ds_names}")
    # logger.info("I'm here")
    #
    # oof_ds.unpersist()

    assert len(persistence_manager.children) == 0
    assert len(persistence_manager.all_datasets) == 2

    oof_ds.unpersist()
    pred_ds.unpersist()

    assert len(persistence_manager.all_datasets) == 0

    logger.info("Finished")
