from pyspark.sql import SparkSession

from .utils import DummyTabularAutoML
from .. import spark as spark_sess

spark = spark_sess


def test_automl_preset(spark: SparkSession):
    train_data = spark.createDataFrame([
        {"a": i, "b": 100 + i, "c": 100 * i, "TARGET": i % 5} for i in range(100)
    ])

    automl = DummyTabularAutoML()

    # 1. check for output result, features, roles (required columns in data, including return_all_predictions)
    # 2. checking for layer-to-layer data transfer (internal in DummyTabularAutoML):
    #   - all predictions of the first level are available in all pipes of the second level
    #   - all inputs data are presented in all pipes of the first level
    #   - all inputs data are presented in all pipes of the second level (if skip_conn)
    # 3. blending and return_all_predictions works correctly
    res_ds = automl.fit_predict(train_data, roles={"target": "TARGET"})
