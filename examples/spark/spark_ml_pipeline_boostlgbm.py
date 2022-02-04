from copy import deepcopy
import logging
import logging.config

from pyspark.ml import Pipeline

from lightautoml.dataset.roles import FoldsRole, TargetRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBMEstimator

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
        sdataset.to_pandas().data.to_csv("/tmp/sdataset_data.csv", index=False)

        feats_to_select_numeric = get_columns_by_role(sdataset, "Numeric")


        # Test BoostLGBM
        boostlgbm_estimator = BoostLGBMEstimator(input_cols=feats_to_select_numeric, 
                                                input_roles=sdataset.roles,
                                                task_name=sdataset.task.name,
                                                folds_column=sdataset.folds_column,
                                                target_column=sdataset.target_column,
                                                folds_number=sreader.cv)

        boostlgbm_pipline = Pipeline(stages=[boostlgbm_estimator])

        boostlgbm_model = boostlgbm_pipline.fit(sdataset.data)

        result = boostlgbm_model.transform(sdataset.data).toPandas().to_csv("/tmp/boostlgbm_model_result.csv", index=False)

        logger.info("Finished")

