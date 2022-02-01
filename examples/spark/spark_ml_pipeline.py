import logging
import logging.config
import os

from lightautoml.dataset.roles import NumericRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.transformers.base import ChangeRoles, UnionTransformer, SequentialTransformer
from lightautoml.spark.transformers.categorical import OrdinalEncoder
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.transformers.base import ColumnsSelector
from lightautoml.spark.transformers.categorical import LabelEncoder, TargetEncoder

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
        sdataset = sreader.fit_read(df, roles=roles)

        # ChangeRoles(output_category_role),
        # feats_to_select_cats = get_columns_by_role(sdataset, "Category", encoding_type="oof")
        # see: lgb_pipeline.py, create_pipeline method, line 189
        feats_to_select_cats = ['back_legroom', 'front_legroom', 'fuel_tank_volume', 'height', 'length',
                                'make_name', 'model_name', 'power', 'torque', 'transmission_display', 'vin',
                                'wheelbase', 'width']
        feats_to_select_ordinal = get_columns_by_role(sdataset, "Category", encoding_type="auto") + get_columns_by_role(
            sdataset, "Category", encoding_type="ohe"
        )
        feats_to_select_numeric = get_columns_by_role(sdataset, "Numeric")

        cat_processing = SequentialTransformer([
            ColumnsSelector(keys=feats_to_select_cats),
            LabelEncoder(random_state=42),
            TargetEncoder()
        ])

        ordinal_cat_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select_ordinal),
                OrdinalEncoder(random_state=42),
            ]
        )

        num_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select_numeric),
                # we don't need this because everything is handled by Spark
                # thus we have no other dataset type except SparkDataset
                # ConvertDataset(dataset_type=NumpyDataset),
                ChangeRoles(NumericRole(np.float32)),
            ]
        )

        ut = UnionTransformer([cat_processing, ordinal_cat_processing, num_processing])

        processed_dataset = ut.fit_transform(sdataset)

        result = processed_dataset.to_pandas().data

        logger.info("Finished")

