import functools
from multiprocessing.pool import ThreadPool
from typing import Tuple

from pandas import DataFrame
from pyspark import inheritable_thread_target
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.sql import functions as sf

from examples.spark.examples_utils import get_spark_session
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.tasks.base import SparkTask


train_features = ['ENTRANCES_MEDI',
 'OWN_CAR_AGE',
 'DAYS_ID_PUBLISH',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'AMT_ANNUITY',
 'AMT_CREDIT',
 'FLAG_DOCUMENT_3',
 'APARTMENTS_MEDI',
 'APARTMENTS_AVG',
 'ord__CODE_GENDER',
 'ord__NAME_TYPE_SUITE',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'REGION_RATING_CLIENT',
 'COMMONAREA_AVG',
 'FLAG_DOCUMENT_19',
 'NONLIVINGAPARTMENTS_MODE',
 'REG_REGION_NOT_LIVE_REGION',
 'COMMONAREA_MODE',
 'LIVINGAPARTMENTS_AVG',
 'TOTALAREA_MODE',
 'FLOORSMAX_MODE',
 'DAYS_LAST_PHONE_CHANGE',
 'FLOORSMIN_AVG',
 'FLAG_EMP_PHONE',
 'ord__NAME_CONTRACT_TYPE',
 'ord__WALLSMATERIAL_MODE',
 'YEARS_BUILD_MEDI',
 'BASEMENTAREA_AVG',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'LIVINGAPARTMENTS_MEDI',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_16',
 'FLOORSMIN_MODE',
 'LANDAREA_MEDI',
 'FLAG_DOCUMENT_8',
 'ord__NAME_INCOME_TYPE',
 'NONLIVINGAREA_MEDI',
 'FLOORSMAX_AVG',
 'LIVINGAREA_MODE',
 'APARTMENTS_MODE',
 'EXT_SOURCE_3',
 'YEARS_BUILD_MODE',
 'NONLIVINGAREA_AVG',
 'YEARS_BEGINEXPLUATATION_MODE',
 'ord__OCCUPATION_TYPE',
 'FLOORSMIN_MEDI',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'REGION_RATING_CLIENT_W_CITY',
 'ord__NAME_EDUCATION_TYPE',
 'EXT_SOURCE_2',
 'FLAG_EMAIL',
 'HOUR_APPR_PROCESS_START',
 'NONLIVINGAPARTMENTS_MEDI',
 'ENTRANCES_MODE',
 'FLAG_DOCUMENT_14',
 'LANDAREA_AVG',
 'ELEVATORS_MODE',
 'ord__FONDKAPREMONT_MODE',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'FLAG_CONT_MOBILE',
 'ord__FLAG_OWN_CAR',
 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_AVG',
 'LANDAREA_MODE',
 'FLAG_DOCUMENT_9',
 'CNT_FAM_MEMBERS',
 'ENTRANCES_AVG',
 'AMT_GOODS_PRICE',
 'ord__WEEKDAY_APPR_PROCESS_START',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'DAYS_REGISTRATION',
 'LIVINGAREA_AVG',
 'LIVE_REGION_NOT_WORK_REGION',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'NONLIVINGAPARTMENTS_AVG',
 'YEARS_BUILD_AVG',
 'REGION_POPULATION_RELATIVE',
 'ord__ORGANIZATION_TYPE',
 'ord__NAME_HOUSING_TYPE',
 'ord__FLAG_OWN_REALTY',
 'LIVINGAPARTMENTS_MODE',
 'FLAG_DOCUMENT_13',
 'FLOORSMAX_MEDI',
 'ord__EMERGENCYSTATE_MODE',
 'AMT_INCOME_TOTAL',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'EXT_SOURCE_1',
 'ELEVATORS_MEDI',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'REG_CITY_NOT_WORK_CITY',
 'ord__NAME_FAMILY_STATUS',
 'NONLIVINGAREA_MODE',
 'REG_CITY_NOT_LIVE_CITY',
 'DAYS_EMPLOYED',
 'LIVE_CITY_NOT_WORK_CITY',
 'LIVINGAREA_MEDI',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_6',
 'COMMONAREA_MEDI',
 'ord__HOUSETYPE_MODE',
 'FLAG_WORK_PHONE',
 'DAYS_BIRTH',
 'ELEVATORS_AVG',
 'FLAG_PHONE',
 'BASEMENTAREA_MODE',
 'CNT_CHILDREN',
 'REG_REGION_NOT_WORK_REGION']


params = {
        'learningRate': 0.01,
        'numLeaves': 32,
        'featureFraction': 0.7,
        'baggingFraction': 0.7,
        'baggingFreq': 1,
        'maxDepth': -1,
        'minGainToSplit': 0.0,
        'maxBin': 255,
        'minDataInLeaf': 5,
        'numIterations': 500,
        'earlyStoppingRound': 200,
        'objective': 'binary',
        'metric': 'auc',
        'rawPredictionCol': 'raw_prediction',
        'probabilityCol': 'LightGBM_prediction_0',
        'predictionCol': 'prediction',
        'isUnbalance': True,
        'numThreads': 3
    }


def train_model(fold:int, train_df: DataFrame, test_df: DataFrame) -> Tuple[int, float]:
    assembler = VectorAssembler(
        inputCols=train_features,
        outputCol=f"LightGBM_vassembler_features",
        handleInvalid="keep"
    )

    lgbm = LightGBMClassifier(
        **params,
        featuresCol=assembler.getOutputCol(),
        labelCol='TARGET',
        validationIndicatorCol='is_val',
        verbosity=1,
        useSingleDatasetMode=True,
        isProvideTrainingMetric=True,
        chunkSize=4_000_000,
    )

    train_df = train_df.withColumn('is_val', sf.col('reader_fold_num') == fold)

    transformer = lgbm.fit(assembler.transform(train_df))
    preds_df = transformer.transform(assembler.transform(test_df))

    score = SparkTask("binary").get_dataset_metric()
    metric_value = score(
        preds_df.select(
            SparkDataset.ID_COLUMN,
            sf.col('TARGET').alias('target'),
            sf.col(params['probabilityCol']).alias('prediction')
        )
    )
    return fold, metric_value


def main():
    spark = get_spark_session()

    train_df = spark.read.parquet("/opt/slama_data/train.parquet")
    test_df = spark.read.parquet("/opt/slama_data/test.parquet")

    tasks =[
        functools.partial(
            train_model,
            fold,
            train_df,
            test_df
        )
        for fold in range(3)
    ]

    pool = ThreadPool(processes=1)
    tasks = map(inheritable_thread_target, tasks)
    results = (result for result in pool.imap_unordered(lambda f: f(), tasks) if result)
    results = sorted(results, key=lambda x: x[0])

    for fold, metric_value in results:
        print(f"Metric value (fold = {fold}): {metric_value}")


if __name__ == "__main__":
    main()