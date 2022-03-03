from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from synapse.ml.lightgbm import LightGBMRegressor

from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator

import numpy as np

spark = SparkSession.builder.getOrCreate()


data = [
    {'_id': i, 'a': i, 'b': i + 10, 'c': i * 10, 'target': i, 'is_val': i % 2} for i in range(100)
]

df = spark.createDataFrame(data)
ds = SparkDataset(df,
                  {'a': CategoryRole(np.float32), 'b': CategoryRole(np.float32), 'c': CategoryRole(np.float32)},
                  task=SparkTask('reg'),
                  target='target')

est = SparkLabelEncoderEstimator(input_cols=ds.features, input_roles=ds.roles)
est.fit(ds.data)

_assembler = VectorAssembler(
    inputCols=['a', 'b', 'c'],
    outputCol=f"vassembler_features",
    handleInvalid="keep"
)

params = {'learningRate': 0.01, 'numLeaves': 32, 'featureFraction': 1, 'baggingFraction': 0.7, 'baggingFreq': 1,
 'maxDepth': -1, 'minGainToSplit': 0.0, 'maxBin': 255, 'minDataInLeaf': 5, 'numIterations': 3000,
 'earlyStoppingRound': 200, 'numThreads': 3, 'objective': 'regression', 'metric': 'mse'}

lgbm = LightGBMRegressor(
    **params,
    featuresCol=_assembler.getOutputCol(),
    labelCol='target',
    validationIndicatorCol='is_val',
    verbosity=1,
    useSingleDatasetMode=True,
    isProvideTrainingMetric=True
)

temp_sdf = _assembler.transform(df)
ml_model = lgbm.fit(temp_sdf)

print("Success")

spark.stop()