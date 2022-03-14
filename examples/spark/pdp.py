import datetime
import logging.config
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyspark.ml import Transformer, PipelineModel
from pyspark.sql import Window, functions as F, types as SparkTypes
from tqdm import tqdm
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


def _get_histogram(data: SparkDataFrame, column: str, n_bins: int) -> Tuple[List, np.ndarray]:
    assert n_bins >= 2, "n_bins must be equal 2 or more"
    bin_edges, counts = data \
        .select(F.col(column).cast("double")) \
        .where(F.col(column).isNotNull()) \
        .rdd.map(lambda x : x[0]) \
        .histogram(n_bins)
    bin_edges = np.array(bin_edges)
    return counts, bin_edges


def get_pdp_data_numeric_feature(df: SparkDataFrame,
                                 feature_name: str,
                                 model: PipelineModel,
                                 prediction_col: str,
                                 n_bins: int) -> Tuple[List, List, List]:
    counts, bin_edges = _get_histogram(df, feature_name, n_bins)
    grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    ys = []
    for i in tqdm(grid):
        # replace feature column values with constant
        sdf = df.select(*[c for c in df.columns if c != feature_name], F.lit(i).alias(feature_name))

        # infer via transformer
        preds = model.transform(sdf)
        preds.show(truncate=False)
        preds = np.array(preds.select(prediction_col).collect())
        ys.append(preds)
    return grid, ys, counts


def get_pdp_data_categorical_feature(df: SparkDataFrame,
                                     feature_name: str,
                                     model: PipelineModel,
                                     prediction_col: str,
                                     n_top_cats: int) -> Tuple[List, List, List]:
    """Returns grid, ys, counts to plot PDP

    Args:
        df (SparkDataFrame): Spark DataFrame with `feature_name` column
        feature_name (str): feature column name
        model (PipelineModel): Spark Pipeline Model
        prediction_col (str): prediction column to be created by the `model`
        n_top_cats (int): param to selection top n categories
    
    Returns:
        Tuple[List, List, List]:
        `grid` is list of categories,
        `ys` is list of predictions by category,
        `counts` is numbers of values by category
    """
    feature_cnt = df.groupBy(feature_name).count().orderBy(F.desc("count")).collect()
    grid = [row[feature_name] for row in feature_cnt[:n_top_cats]]
    counts = [row["count"] for row in feature_cnt[:n_top_cats]]
    ys = []
    for i in tqdm(grid):
        sdf = df.select(*[c for c in df.columns if c != feature_name], F.lit(i).alias(feature_name))
        preds = model.transform(sdf)
        preds = np.array(preds.select(prediction_col).collect())
        ys.append(preds)
    if len(feature_cnt) > n_top_cats:

        # unique other categories
        unique_other_categories = [row[feature_name] for row in feature_cnt[n_top_cats:]]

        # get non-top categories, natural distributions is important here
        w = Window().orderBy(F.lit('A'))  # window without sorting
        other_categories_collection = df.select(feature_name) \
            .filter(F.col(feature_name).isin(unique_other_categories)) \
            .select(F.row_number().over(w).alias("row_num"), feature_name) \
            .collect()
        
        # dict with key=%row number% and value=%category%
        other_categories_dict = {x["row_num"]: x[feature_name] for x in other_categories_collection}
        max_row_num = len(other_categories_collection)

        def get_category_by_row_num(row_num):
            if (remainder := row_num % max_row_num) == 0:
                key = row_num
            else:
                key = remainder
            return other_categories_dict[key]
        get_category_udf = F.udf(get_category_by_row_num, SparkTypes.StringType())

        # add row number to main dataframe and exclude feature_name column
        sdf = df.select(F.row_number().over(w).alias("row_num"), *[f for f in df.columns if f != feature_name])

        all_columns_except_row_num = [f for f in sdf.columns if f != "row_num"]
        feature_col = get_category_udf(F.col("row_num")).alias(feature_name)
        # exclude row number from dataframe
        # and add back feature_name column filled with other categories same distribution
        sdf = sdf.select(*all_columns_except_row_num, feature_col)

        preds = model.transform(sdf)
        preds = np.array(preds.select(prediction_col).collect())

        grid.append("<OTHER>")
        ys.append(preds)
        counts.append(sum([row["count"] for row in feature_cnt[n_top_cats:]]))

    return grid, ys, counts


def get_pdp_data_datetime_feature(df: SparkDataFrame,
                                  feature_name: str,
                                  model: PipelineModel,
                                  prediction_col: str,
                                  datetime_level: str) -> Tuple[List, List, List]:
    # test_data_read = self.reader.read(df)
    if datetime_level == "year":
        feature_cnt = df.groupBy(F.year(feature_name).alias("year")).count().orderBy(F.asc("year")).collect()
        grid = [x["year"] for x in feature_cnt]
        counts = [row["count"] for row in feature_cnt]
        replace_date_element_udf = F.udf(replace_year_in_date, SparkTypes.DateType())
    elif datetime_level == "month":
        feature_cnt = df.groupBy(F.month(feature_name).alias("month")).count().orderBy(F.asc("month")).collect()
        grid = np.arange(1, 13)
        grid = grid.tolist()
        counts = [0] * 12
        for row in feature_cnt:
            counts[row["month"]-1] = row["count"]
        replace_date_element_udf = F.udf(replace_month_in_date, SparkTypes.DateType())
    else:
        feature_cnt = df.groupBy(F.dayofweek(feature_name).alias("dayofweek")).count().orderBy(F.asc("dayofweek")).collect()
        grid = np.arange(7)
        grid = grid.tolist()
        counts = [0] * 7
        for row in feature_cnt:
            counts[row["dayofweek"]-1] = row["count"]
        replace_date_element_udf = F.udf(replace_dayofweek_in_date, SparkTypes.DateType())
    ys = []
    for i in tqdm(grid):
        all_columns_except_feature = [c for c in df.columns if c != feature_name]
        feature_col = replace_date_element_udf(F.col(feature_name), F.lit(i)).alias(feature_name)
        sdf = df.select(*all_columns_except_feature, feature_col)
        preds = model.transform(sdf)
        preds.show(truncate=False)
        preds = np.array(preds.select(prediction_col).collect())
        ys.append(preds)

    return grid, ys, counts


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
        # get_pdp_data_categorical_feature(df, "category_feature", pipeline_model, "prediction", 2)

        # get_pdp_data_numeric_feature(df, "numeric_feature", pipeline_model, "prediction", 2)

        # get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "year")
        get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "month")
        # get_pdp_data_datetime_feature(df, "datetime_feature", pipeline_model, "prediction", "dayofweek")


        logger.info("Finished")
