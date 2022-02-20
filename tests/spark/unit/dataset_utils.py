import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pyspark.sql import SparkSession

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask


DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


def dump_data(path: str, ds: SparkDataset, **meta_kwargs):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    metadata = {
        "roles": ds.roles,
        "target": ds.target_column,
        "folds": ds.folds_column,
        "task_name": ds.task.name if ds.task else None
    }
    metadata.update(meta_kwargs)

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    ds.data.write.parquet(data_file)


def load_dump_if_exist(spark: SparkSession, path: Optional[str] = None) -> Optional[Tuple[SparkDataset, Dict]]:
    if path is None:
        return None

    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file).repartition(16).cache()
    df.write.mode('overwrite').format('noop').save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"]
    )

    return ds, metadata


def datasets() -> Dict[str, Any]:
    all_datastes = {
        "used_cars_dataset": {
            "path": "/opt/spark_data/small_used_cars_data.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_head50k": {
            "path": "/opt/spark_data/head50k_0125x_cleaned.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_0125x": {
            "path": "/opt/spark_data/0125x_cleaned.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_025x": {
            "path": "/opt/spark_data/derivative_datasets/025x_cleaned.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_05x": {
            "path": "/opt/spark_data/derivative_datasets/05x_dataset.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_1x": {
            "path": "/opt/spark_data/derivative_datasets/1x_dataset.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "used_cars_dataset_2x": {
            "path": "/opt/spark_data/derivative_datasets/2x_cleaned.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "lama_test_dataset": {
            "path": "/opt/spark_data/sampled_app_train.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "TARGET",
            "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
        },

        # https://www.openml.org/d/734
        "ailerons_dataset": {
            "path": "/opt/spark_data/ailerons.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "binaryClass",
            "roles": {"target": "binaryClass"},
        },

        # https://www.openml.org/d/4534
        "phishing_websites_dataset": {
            "path": "/opt/spark_data/PhishingWebsites.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Result",
            "roles": {"target": "Result"},
        },

        # https://www.openml.org/d/981
        "kdd_internet_usage": {
            "path": "/opt/spark_data/kdd_internet_usage.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Who_Pays_for_Access_Work",
            "roles": {"target": "Who_Pays_for_Access_Work"},
        },

        # https://www.openml.org/d/42821
        "nasa_dataset": {
            "path": "/opt/spark_data/nasa_phm2008.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "class",
            "roles": {"target": "class"},
        },

        # https://www.openml.org/d/4549
        "buzz_dataset": {
            "path": "/opt/spark_data/Buzzinsocialmedia_Twitter_25k.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "Annotation",
            "roles": {"target": "Annotation"},
        },

        # https://www.openml.org/d/372
        "internet_usage": {
            "path": "/opt/spark_data/internet_usage.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Actual_Time",
            "roles": {"target": "Actual_Time"},
        },

        # https://www.openml.org/d/4538
        "gesture_segmentation": {
            "path": "/opt/spark_data/gesture_segmentation.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Phase",
            "roles": {"target": "Phase"},
        },

        # https://www.openml.org/d/382
        "ipums_97": {
            "path": "/opt/spark_data/ipums_97.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "movedin",
            "roles": {"target": "movedin"},
        }
    }

    return all_datastes


def prepared_datasets(spark: SparkSession,
                      cv: int,
                      ds_configs: List[Dict[str, Any]],
                      checkpoint_dir: Optional[str] = None) -> List[SparkDataset]:
    sds = []
    for config in ds_configs:
        path = config['path']
        task_type = config['task_type']
        roles = config['roles']

        ds_name = os.path.basename(os.path.splitext(path)[0])

        dump_path = os.path.join(checkpoint_dir, f"dump_{ds_name}_{cv}.dump") \
            if checkpoint_dir is not None else None
        res = load_dump_if_exist(spark, dump_path)
        if res:
            dumped_ds, _ = res
            sds.append(dumped_ds)
            continue

        df = spark.read.csv(path, header=True, escape="\"")
        sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv)
        sdataset = sreader.fit_read(df, roles=roles)

        if dump_path is not None:
            dump_data(dump_path, sdataset, cv=cv)

        sds.append(sdataset)

    return sds


def get_test_datasets(setting: str = "all") -> List[Dict[str, Any]]:
    dss = datasets()

    if setting == "fast":
        return [dss['used_cars_dataset']]
    elif setting == "multiclass":
        return [dss['internet_usage'], dss['gesture_segmentation']]
    elif setting == "all-tasks":
        return [
            # dss['used_cars_dataset'],
            dss["buzz_dataset"],
            dss['lama_test_dataset'],
            dss["ailerons_dataset"],
            dss["internet_usage"],
            dss["gesture_segmentation"]
        ]
    elif setting == "all":
        # exccluding all heavy datasets
        return list(cfg for ds_name, cfg in dss.items() if not ds_name.startswith('used_cars_dataset_'))
    else:
        raise ValueError(f"Unsupported setting {setting}")


def get_prepared_test_datasets(spark: Union[SparkSession, Callable[[], SparkSession]],
                               cv: int,
                               dss_setting: str = "all",
                               checkpoint_dir: Optional[str] = None) -> List[SparkDataset]:
    ds_configs = get_test_datasets(dss_setting)

    if not isinstance(spark, SparkSession):
        spark = spark()

    sdss = prepared_datasets(spark, cv, ds_configs, checkpoint_dir)

    return sdss

