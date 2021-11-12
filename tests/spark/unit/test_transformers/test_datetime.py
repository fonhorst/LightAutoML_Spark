import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.spark.transformers.datetime import TimeToNum as SparkTimeToNum
from lightautoml.transformers.datetime import TimeToNum

from . import compare_by_content, compare_by_metadata, spark


def test_time_to_num(spark: SparkSession):

    source_data = pd.DataFrame(data={
        "night": [
            "2000-01-01 00:00:00",
            np.nan,
            "2020-01-01 00:00:00",
            "2025-01-01 00:00:00",
            "2100-01-01 00:00:00",
        ],
        "morning": [
            "2000-01-01 06:00:00",
            "2017-01-01 06:00:00",
            "2020-01-01 06:00:00",
            None,
            "2100-01-01 06:00:00",
        ],
        "day": [
            np.nan,
            "2017-01-01 12:00:00",
            "2020-01-01 12:00:00",
            "2025-01-01 12:00:00",
            "2100-01-01 12:00:00",
        ],
        "evening": [
            "2000-01-01 20:00:00",
            "2017-01-01 20:00:00",
            "2020-01-01 20:00:00",
            "2025-01-01 20:00:00",
            "2100-01-01 20:00:00",
        ],
    })

    ds = PandasDataset(source_data, roles={name: DatetimeRole() for name in source_data.columns})

    compare_by_content(spark, ds, TimeToNum(), SparkTimeToNum())

