import logging.config
import os
from typing import Sequence, Optional, Iterable
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML, ReadableIntoSparkDf
from lightautoml.spark.dataset.base import SparkDataFrame
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.report import ReportDeco
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def get_spark_session():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        return SparkSession.builder.getOrCreate()

    spark_sess = (
        SparkSession
        .builder
        .master("local[5]")
        .config("spark.jars", "/home/nikolay/wspace/LightAutoML/jars/spark-lightautoml_2.12-0.1.jar")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    return spark_sess


class SparkCachingTabularMLWrapper:
    def __init__(self, automl: SparkTabularAutoML):
        self._automl = automl
        self._oof_preds = None
        self._preds = None

    @property
    def reader(self):
        return self._automl.reader

    def fit_predict(
            self,
            train_data: ReadableIntoSparkDf,
            roles: Optional[dict] = None,
            train_features: Optional[Sequence[str]] = None,
            cv_iter: Optional[Iterable] = None,
            valid_data: Optional[ReadableIntoSparkDf] = None,
            valid_features: Optional[Sequence[str]] = None,
            log_file: str = None,
            verbose: int = 0,
    ) -> SparkDataset:
        if self._oof_preds is None:
            self._oof_preds = self._automl.fit_predict(train_data, roles, train_features,
                                                       cv_iter, valid_data, valid_features, log_file, verbose)
            new_ds = self._oof_preds.empty()
            new_ds.set_data(self._oof_preds.data.cache(), self._oof_preds.features, self._oof_preds.roles)
            self._oof_preds = new_ds

        return self._oof_preds

    def predict(
            self,
            data: ReadableIntoSparkDf,
            features_names: Optional[Sequence[str]] = None,
            return_all_predictions: Optional[bool] = None,
            add_reader_attrs: bool = False
    ) -> SparkDataset:
        if self._preds is None:
            self._preds = self._automl.predict(data, features_names, return_all_predictions, add_reader_attrs)

            new_ds = self._preds.empty()
            new_ds.set_data(self._preds.data.cache(), self._preds.features, self._preds.roles)
            self._preds = new_ds

        return self._preds

    def get_feature_scores(
            self,
            calc_method: str = "fast",
            data: Optional[ReadableIntoSparkDf] = None,
            features_names: Optional[Sequence[str]] = None,
            silent: bool = True,
    ):
        return self._automl.get_feature_scores(calc_method, data, features_names, silent)

    def get_individual_pdp(
            self,
            test_data: SparkDataFrame,
            feature_name: str,
            n_bins: Optional[int] = 30,
            top_n_categories: Optional[int] = 10,
            datetime_level: Optional[str] = "year",
            ice_fraction: float = 1.0,
            ice_fraction_seed: int = 42
    ):
        return self._automl.get_individual_pdp(test_data, feature_name, n_bins,
                                               top_n_categories, datetime_level, ice_fraction, ice_fraction_seed)


# if __name__ == "__main__":
spark = get_spark_session()
spark.sparkContext.setLogLevel("ERROR")

seed = 42
cv = 2
# use_algos = [["lgb", "linear_l2"], ["lgb"]]
# use_algos = [["linear_l2"]]
use_algos = [["lgb"]]
task_type = "binary"
roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}

DATASET_DIR = '/tmp/'
DATASET_NAME = 'sampled_app_train.csv'
DATASET_FULLNAME = os.path.join(DATASET_DIR, DATASET_NAME)
DATASET_URL = 'https://raw.githubusercontent.com/sberbank-ai-lab/LightAutoML/master/examples/data/sampled_app_train.csv'

if not os.path.exists(DATASET_FULLNAME):
    os.makedirs(DATASET_DIR, exist_ok=True)

    dataset = requests.get(DATASET_URL).text
    with open(DATASET_FULLNAME, 'w') as output:
        output.write(dataset)

data = pd.read_csv(DATASET_FULLNAME)
data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                    ).astype(str)

data.to_csv("/tmp/sampled_app_train.csv", index=False)

train_data, test_data = prepare_test_and_train(spark, "/tmp/sampled_app_train.csv", seed)

with log_exec_timer("spark-lama training") as train_timer:
    task = SparkTask(task_type)

    automl = SparkTabularAutoML(
        spark=spark,
        task=task,
        lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 500}},
        linear_l2_params={"default_params": {"regParam": [1]}},
        general_params={"use_algos": use_algos},
        reader_params={"cv": cv, "advanced_roles": False, 'random_state': seed}
    )

    c_automl = SparkCachingTabularMLWrapper(automl)
    oof_preds = c_automl.fit_predict(train_data, roles=roles, valid_data=test_data)

    pred_col = next(c for c in oof_preds.data.columns if c.startswith('prediction'))
    score = task.get_dataset_metric()(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        F.col(oof_preds.target_column).alias('target'),
        F.col(pred_col).alias('prediction')
    ))
    print(f"Score: {score}")

    report_automl = ReportDeco(
        output_path="/tmp/",
        report_file_name="spark_lama_report.html",
        interpretation=True
    )(c_automl)

    report_automl.fit_predict(train_data, roles=roles)
    report_automl.predict(test_data, add_reader_attrs=True)