from copy import deepcopy
import logging
import logging.config
from typing import cast

from pyspark.ml import Pipeline

from lightautoml.dataset.roles import FoldsRole, TargetRole
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM as SparkBoostLGBM

import numpy as np

from lightautoml.validation.base import DummyIterator, HoldoutIterator

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

        feats_to_select_numeric = get_columns_by_role(sdataset_tmp, "Numeric")
        
        sdataset = sdataset_tmp.empty()
        sdataset.set_data(
            sdataset_tmp.data
                .select(SparkDataset.ID_COLUMN, *feats_to_select_numeric)
                .join(sdataset.target, on=SparkDataset.ID_COLUMN)
                .join(sdataset.folds, on=SparkDataset.ID_COLUMN),
            sdataset_tmp.features,
            sdataset_tmp.roles
        )
        # new_roles = deepcopy(sdataset_tmp.roles)
        # new_roles.update({sdataset_tmp.target_column: TargetRole(np.float32), sdataset_tmp.folds_column: FoldsRole()})
        # sdataset.set_data(
        #     sdataset_tmp.data \
        #         .join(sdataset_tmp.target, SparkDataset.ID_COLUMN) \
        #         .join(sdataset_tmp.folds, SparkDataset.ID_COLUMN),
        #     sdataset_tmp.features,
        #     new_roles
        # )
        # sdataset.to_pandas().data.to_csv("/tmp/sdataset_data.csv", index=False)

        # Test SparkBoostLGBM and SparkBoostLGBM Transformer
        iterator = DummyIterator(sdataset)
        spark_ml_algo = SparkBoostLGBM()
        spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)
        spark_ml_algo = cast(SparkBoostLGBM, spark_ml_algo)
        sdf = spark_ml_algo.transformer.transform(sdataset.data)

        sdf.write.mode('overwrite').format('noop').save()

        logger.info("Finished")
