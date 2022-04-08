from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs

from lightautoml.spark.automl.base import SparkAutoML
from lightautoml.spark.automl.blend import SparkWeightedBlender
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator, ImportanceCutoffSelector
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.utils.timer import PipelineTimer


if __name__ == "__main__":
    spark = get_spark_session()

    dataset_path, task_type, roles, dtype = get_dataset_attrs("lama_test_dataset")

    train, test = prepare_test_and_train(spark, dataset_path, seed=42)

    cacher_key = "main_cache"


    # ======================================================================================
    print("Create timer...")
    timer = PipelineTimer(900, mode=2)
    print("Timer created...")
    # ======================================================================================
    print("Create selector...")
    timer_gbm = timer.get_task_timer("gbm")
    feat_sel_0 = SparkLGBSimpleFeatures(cacher_key=cacher_key)

    mod_sel_0 = SparkBoostLGBM(
        timer=timer_gbm,
        cacher_key=cacher_key
    )

    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(
        feat_sel_0,
        mod_sel_0,
        imp_sel_0,
        cutoff=0,
    )
    print("Selector created...")
    # ======================================================================================
    print("Create gbms...")
    feats_gbm_0 = SparkLGBAdvancedPipeline(
        top_intersections=4,
        output_categories=True,
        feats_imp=imp_sel_0,
        cacher_key=cacher_key
    )
    timer_gbm_0 = timer.get_task_timer("gbm")
    timer_gbm_1 = timer.get_task_timer("gbm")

    gbm_0 = SparkBoostLGBM(timer=timer_gbm_0, cacher_key=cacher_key)
    gbm_1 = SparkBoostLGBM(timer=timer_gbm_1, cacher_key=cacher_key)

    tuner_0 = OptunaTuner(n_trials=10, timeout=10, fit_on_holdout=True)
    gbm_lvl0 = SparkMLPipeline(
        cacher_key=cacher_key,
        ml_algos=[(gbm_0, tuner_0), gbm_1],
        pre_selection=selector_0,
        features_pipeline=feats_gbm_0,
        post_selection=None
    )
    print("Gbms created...")
    # ======================================================================================
    print("Create linear...")
    feats_reg_0 = SparkLinearFeatures(output_categories=True, sparse_ohe="auto", cacher_key=cacher_key)

    timer_reg = timer.get_task_timer("reg")
    reg_0 = SparkLinearLBFGS(cacher_key=cacher_key, timer=timer_reg)

    reg_lvl0 = SparkMLPipeline(
        cacher_key=cacher_key,
        ml_algos=[reg_0],
        pre_selection=None,
        features_pipeline=feats_reg_0,
        post_selection=None
    )
    print("Linear created...")
    # ======================================================================================
    print("Create reader...")
    task = SparkTask(task_type)
    reader = SparkToSparkReader(task, advanced_roles=False)
    print("Reader created...")
    # ======================================================================================
    print("Create blender...")
    blender = SparkWeightedBlender()
    print("Blender created...")
    # ======================================================================================
    print("Create AutoML...")
    automl = SparkAutoML(
        reader=reader,
        levels=[[gbm_lvl0, reg_lvl0]],
        timer=timer,
        blender=blender,
        skip_conn=False,
    )
    print("AutoML created...")
    # ======================================================================================
    print("Fit predict...")
    oof_pred = automl.fit_predict(train, roles={"target": "TARGET"})
    print("Finnished fitting...")
    score = task.get_dataset_metric()
    metric_value = score(oof_pred)
    print(f"Score for out-of-fold predictions: {metric_value}")

    test_pred = automl.predict(test, add_reader_attrs=True)
    score = task.get_dataset_metric()
    metric_value = score(test_pred)
    print(f"Score for test predictions: {metric_value}")


