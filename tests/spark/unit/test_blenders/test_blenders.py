import random

from lightautoml.dataset.roles import NumericRole
from pyspark.sql import SparkSession

from sparklightautoml.automl.blend import SparkWeightedBlender
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import log_exec_time
from sparklightautoml.validation.iterators import SparkDummyIterator
from .. import spark as spark_sess
from ..test_auto_ml.utils import DummyMLAlgo

spark = spark_sess


# noinspection PyShadowingNames
def test_weighted_blender(spark: SparkSession):
    target_col = "some_target"
    folds_col = "folds"
    n_classes = 10
    models_count = 4
    persistence_manager = PlainCachePersistenceManager()

    data = [
        {
            SparkDataset.ID_COLUMN: i,
            "a": i, "b": 100 + i, "c": 100 * i,
            target_col: random.randint(0, n_classes),
            folds_col: random.randint(0, 2)
        }
        for i in range(100)
    ]

    roles = {"a": NumericRole(), "b": NumericRole(), "c": NumericRole()}

    data_sdf = spark.createDataFrame(data)
    data_sds = SparkDataset(
        data=data_sdf,
        task=SparkTask("multiclass"),
        persistence_manager=persistence_manager,
        roles=roles,
        target=target_col,
        folds=folds_col,
        name="WeightedBlenderData"
    )

    pipes = [
        SparkMLPipeline(ml_algos=[DummyMLAlgo(n_classes, name=f"dummy_0_{i}")])
        for i in range(models_count)
    ]

    for pipe in pipes:
        data_sds = pipe.fit_predict(SparkDummyIterator(data_sds))

    preds_roles = {c: role for c, role in data_sds.roles.items() if c not in roles}

    sdf = data_sds.data.drop(*list(roles.keys())).cache()
    sdf.write.mode('overwrite').format('noop').save()
    ml_ds = data_sds.empty()
    ml_ds.set_data(sdf, list(preds_roles.keys()), preds_roles, name=data_sds.name)

    swb = SparkWeightedBlender(max_iters=1, max_inner_iters=1)
    with log_exec_time('Blender fit_predict'):
        blended_sds, filtered_pipes = swb.fit_predict(ml_ds, pipes)
        blended_sds.data.write.mode('overwrite').format('noop').save()

    with log_exec_time('Blender predict'):
        transformed_preds_sdf = swb.transformer.transform(ml_ds.data)
        transformed_preds_sdf.write.mode('overwrite').format('noop').save()

    assert len(swb.output_roles) == 1
    prediction, role = list(swb.output_roles.items())[0]
    if data_sds.task.name in ["binary", "multiclass"]:
        assert isinstance(role, NumericVectorOrArrayRole)
    else:
        assert isinstance(role, NumericRole)
    assert prediction in blended_sds.data.columns
    assert prediction in blended_sds.roles
    assert blended_sds.roles[prediction] == role
    assert prediction in transformed_preds_sdf.columns
