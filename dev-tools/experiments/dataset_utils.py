from typing import Any, Dict


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

        "used_cars_dataset_cats_only": {
            "path": "/opt/spark_data/small_used_cars_data.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": [
                    'back_legroom', 'listed_date', 'seller_rating', 'major_options',
                    'exterior_color', 'model_name', 'combine_fuel_economy', 'front_legroom',
                    'width', 'horsepower', 'city', 'trim_name', 'height', 'length',
                    'bed_length', 'wheelbase', 'daysonmarket', 'latitude', 'power',
                    'year', 'maximum_seating', 'savings_amount', 'highway_fuel_economy',
                    'owner_count', 'mileage', 'description', 'engine_displacement',
                    'torque', 'main_picture_url', 'trimId', 'vin', 'vehicle_damage_category',
                    'sp_name', 'dealer_zip', 'listing_id', 'longitude', 'bed_height',
                    'fuel_tank_volume', 'interior_color', 'is_certified', 'sp_id',
                    'city_fuel_economy'
                ],
                "category": [
                    'is_cpo', 'franchise_make', 'transmission_display', 'theft_title',
                    'isCab', 'franchise_dealer', 'transmission', 'wheel_system',
                    'make_name', 'cabin', 'bed', 'engine_type', 'listing_color',
                    'is_new', 'has_accidents', 'frame_damaged', 'fuel_type',
                    'wheel_system_display', 'salvage', 'fleet', 'body_type',
                    'is_oemcpo', 'engine_cylinders'
                ]
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "tiny_used_cars_dataset": {
            "path": "/opt/spark_data/tiny_used_cars_data_cleaned.csv",
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

        "used_cars_dataset_head60k": {
            "path": "/opt/spark_data/head60k_0125x_cleaned.csv",
            "train_path": "/opt/spark_data/head60k_0125x_cleaned_train.csv",
            "test_path": "/opt/spark_data/head60k_0125x_cleaned_test.csv",
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

        "used_cars_dataset_head2x60k": {
            "path": "/opt/spark_data/head2x60k_0125x_cleaned.csv",
            # "train_path": "/opt/spark_data/head60k_0125x_cleaned_train.csv",
            # "test_path": "/opt/spark_data/head60k_0125x_cleaned_test.csv",
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

        "used_cars_dataset_head62_5k": {
            "path": "/opt/spark_data/head62_5k_0125x_cleaned.csv",
            "train_path": "/opt/spark_data/head62_5k_0125x_cleaned_train.csv",
            "test_path": "/opt/spark_data/head62_5k_0125x_cleaned_test.csv",
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

        "used_cars_dataset_head100k": {
            "path": "/opt/spark_data/head100k_0125x_cleaned.csv",
            "train_path": "/opt/spark_data/head100k_0125x_cleaned_train.csv",
            "test_path": "/opt/spark_data/head100k_0125x_cleaned_test.csv",
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

        "used_cars_dataset_head65k": {
            "path": "/opt/spark_data/head65k_0125x_cleaned.csv",
            "train_path": "/opt/spark_data/head65k_0125x_cleaned_train.csv",
            "test_path": "/opt/spark_data/head65k_0125x_cleaned_test.csv",
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

        "used_cars_dataset_head70k": {
            "path": "/opt/spark_data/head70k_0125x_cleaned.csv",
            "train_path": "/opt/spark_data/head70k_0125x_cleaned_train.csv",
            "test_path": "/opt/spark_data/head70k_0125x_cleaned_test.csv",
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
            "path": "/opt/spark_data/derivative_datasets/0125x_cleaned.csv",
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

        "used_cars_dataset_1x_original": {
            "path": "/opt/spark_data/used_cars_data.csv",
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

        "used_cars_dataset_4x": {
            "path": "/opt/spark_data/derivative_datasets/4x_cleaned.csv",
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
