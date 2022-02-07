from copy import deepcopy
import logging
import logging.config
import os
from typing import List, cast

from pyspark.ml import Pipeline, PipelineModel

from lightautoml.dataset.roles import FoldsRole, NumericRole, TargetRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import SparkMLEstimatorWrapper
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.transformers.base import ChangeRoles, ChangeRolesTransformer, UnionTransformer, SequentialTransformer
from lightautoml.spark.transformers.categorical import LabelEncoderEstimator, OrdinalEncoder, OrdinalEncoderEstimator, TargetEncoderEstimator
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.transformers.base import ColumnsSelector
from lightautoml.spark.transformers.categorical import LabelEncoder, TargetEncoder

import numpy as np

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def execute_lama_pipeline(sdataset: SparkDataset, cat_feats: List[str], ordinal_feats: List[str], numeric_feats: List[str]):
    cat_processing = SequentialTransformer([
        ColumnsSelector(keys=cat_feats),
        LabelEncoder(random_state=42),
        TargetEncoder()
    ])

    ordinal_cat_processing = SequentialTransformer(
        [
            ColumnsSelector(keys=ordinal_feats),
            OrdinalEncoder(random_state=42),
        ]
    )

    num_processing = SequentialTransformer(
        [
            ColumnsSelector(keys=numeric_feats),
            # we don't need this because everything is handled by Spark
            # thus we have no other dataset type except SparkDataset
            # ConvertDataset(dataset_type=NumpyDataset),
            ChangeRoles(NumericRole(np.float32)),
        ]
    )

    ut = UnionTransformer([cat_processing, ordinal_cat_processing, num_processing])

    processed_dataset = ut.fit_transform(sdataset)

    result = processed_dataset.to_pandas().data


def execute_sparkml_pipeline(sdataset: SparkDataset, cat_feats: List[str], ordinal_feats: List[str], numeric_feats: List[str]):
    le_estimator = LabelEncoderEstimator(input_cols=cat_feats, input_roles=sdataset.roles)
    te_estimator = TargetEncoderEstimator(input_cols=le_estimator.getOutputCols(), 
                                          input_roles=le_estimator.getOutputRoles(),
                                          task_name=sdataset.task.name,
                                          folds_column=sdataset.folds_column,
                                          target_column=sdataset.target_column)
    ord_estimator = OrdinalEncoderEstimator(input_cols=ordinal_feats, input_roles=te_estimator.getOutputRoles())
    num_transformer = ChangeRolesTransformer(roles=NumericRole(np.float32), 
                                            input_cols=numeric_feats, 
                                            input_roles=ord_estimator.getOutputRoles())

    cat_processing = Pipeline(stages=[le_estimator, te_estimator])
    ordinal_cat_processing = Pipeline(stages=[ord_estimator])
    num_processing = Pipeline(stages=[num_transformer])


    ut = Pipeline(stages=[cat_processing, ordinal_cat_processing, num_processing])
    
    ut.write().overwrite().save("/tmp/ut_pipline")

    ut_transformer: PipelineModel = ut.fit(sdataset.data)

    ut_transformer.write().overwrite().save("/tmp/ut_pipline")

    result = ut_transformer.transform(sdataset.data).toPandas()

    k = 0


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
        # sdataset.to_pandas().data.to_csv("/tmp/sdataset_data.csv")

        # ChangeRoles(output_category_role),
        # feats_to_select_cats = get_columns_by_role(sdataset, "Category", encoding_type="oof")
        # see: lgb_pipeline.py, create_pipeline method, line 189
        feats_to_select_cats = ['back_legroom', 'front_legroom', 'fuel_tank_volume', 'height', 'length',
                                'make_name', 'model_name']
        feats_to_select_ordinal = get_columns_by_role(sdataset, "Category", encoding_type="auto") + get_columns_by_role(
            sdataset, "Category", encoding_type="ohe"
        )
        feats_to_select_numeric = get_columns_by_role(sdataset, "Numeric")

        # TODO: 1. determine input and output cols in pipelines
        # TODO: 2. transformers only appends new columns
        # TODO: 3. union finishes with subselecting only output columns
        # TODO: (but blocking of columns still necessary)

        # execute_lama_pipeline(sdataset, feats_to_select_cats, feats_to_select_ordinal, feats_to_select_numeric)

        execute_sparkml_pipeline(sdataset, feats_to_select_cats, feats_to_select_ordinal, feats_to_select_numeric)

        logger.info("Finished")

