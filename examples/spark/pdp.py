import datetime
import logging.config

from pyspark.ml import Transformer, PipelineModel
from pyspark.sql import functions as F
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.automl.presets.utils import replace_dayofweek_in_date, replace_month_in_date, replace_year_in_date
from lightautoml.spark.dataset.base import SparkDataFrame


from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import logging_config
from lightautoml.spark.utils import spark_session


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class ConstantTransformer(Transformer):

    def _transform(self, df: SparkDataFrame):
        return df.select(*df.columns, F.lit(1).alias("prediction"))


if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:

        data = [
            (datetime.datetime(1984, 6, 1, 0, 0), 1.0, "red"),
            (datetime.datetime(1999, 4, 1, 0, 0), 7.0, "black"),
            (datetime.datetime(2021, 12, 1, 0, 0), 1.0, "white"),
            (datetime.datetime(2010, 2, 1, 0, 0), None, "yellow"),
            (datetime.datetime(2003, 1, 1, 0, 0), 5.0, "green"),
            (datetime.datetime(2010, 2, 1, 0, 0), 3.0, "blue"),
            (datetime.datetime(2010, 11, 1, 0, 0), None, "purple"),
            (datetime.datetime(2010, 10, 1, 0, 0), 2.0, "green"),
            (datetime.datetime(2010, 2, 1, 0, 0), 1.0, "red"),
            (datetime.datetime(2008, 2, 4, 0, 0), 5.2, None)
        ]

        pred_cols = ["datetime_feature", "numeric_feature", "category_feature"]
        df = spark.createDataFrame(data, pred_cols)
        logger.info("All source data:")
        df.show(truncate=False)

        # ===========================================
        pipeline_model = PipelineModel(stages=[ConstantTransformer()])
        # SparkTabularAutoML.get_pdp_data_categorical_feature(df, "category_feature", pipeline_model, "prediction", 2)

        # SparkTabularAutoML.get_pdp_data_numeric_feature(df, "numeric_feature", pipeline_model, "prediction", 2)

        SparkTabularAutoML.get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "year")
        # SparkTabularAutoML.get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "month")
        # SparkTabularAutoML.get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "dayofweek")

        logger.info("Finished")
