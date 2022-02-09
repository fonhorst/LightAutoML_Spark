import itertools
from copy import deepcopy
import logging
import logging.config
import os
from typing import List, cast, Tuple, Dict

from pyspark.ml import Pipeline, PipelineModel, Transformer, Estimator

from lightautoml.dataset.roles import FoldsRole, TargetRole
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBAdvancedPipelineTmp, LGBSimpleFeatures, LGBSimpleFeaturesTmp
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.transformers.base import SparkTransformer, SequentialTransformer, UnionTransformer
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.tasks.base import Task as SparkTask

import numpy as np
import toposort

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class NoOpTransformer(Transformer):
    def _transform(self, dataset):
        return dataset


class Cacher(Estimator):
    _cacher_dict: Dict[str, SparkDataFrame] = dict()

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def _fit(self, dataset):
        ds = dataset.cache()
        ds.write.mode('overwrite').format('noop').save()

        previous_ds = self._cacher_dict.get(self._key, None)
        if previous_ds:
            previous_ds.unpersist()

        self._cacher_dict[self._key] = ds

        return NoOpTransformer()


def build_graph(begin: SparkTransformer):
    graph = dict()
    def find_start_end(tr: SparkTransformer) -> Tuple[List[SparkTransformer], List[SparkTransformer]]:
        if isinstance(tr, SequentialTransformer):
            se = [st_or_end for el in tr.transformer_list for st_or_end in find_start_end(el)]

            starts = se[0]
            ends = se[-1]
            middle = se[1:-1]

            i = 0
            while i < len(middle):
                for new_st, new_end in itertools.product(middle[i], middle[i + 1]):
                    if new_end not in graph:
                        graph[new_end] = set()
                    graph[new_end].add(new_st)
                i += 2

            return starts, ends

        elif isinstance(tr, UnionTransformer):
            se = [find_start_end(el) for el in tr.transformer_list]
            starts = [s_el for s, _ in se for s_el in s]
            ends = [e_el for _, e in se for e_el in e]
            return starts, ends
        else:
            return [tr], [tr]

    starts, _ = find_start_end(begin)

    for st in starts:
        if st not in graph:
            graph[st] = set()
        graph[st].add("init")

    return graph


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
        df = spark.read.csv("../data/tiny_used_cars_data.csv", header=True, escape="\"")
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

        ml_alg_kwargs = {
            'auto_unique_co': 10,
            'max_intersection_depth': 3,
            'multiclass_te_co': 3,
            'output_categories': False,
            # 'subsample': 100000,
            'top_intersections': 4
        }

        # SparkTransformer pipeline
        lgb_simple_feature_builder = LGBAdvancedPipeline(**ml_alg_kwargs)
        pipeline = lgb_simple_feature_builder.create_pipeline(sdataset)
        # pipeline.fit_transform(sdataset).to_pandas().data.to_csv("/tmp/pipeline_model.csv", index=False)

        graph = build_graph(pipeline)
        tr_layers = list(toposort.toposort(graph))
        stages = [tr for layer in tr_layers
                  for tr in itertools.chain(layer, [Cacher('some_key')])]

        spark_ml_pipeline = Pipeline(stages=stages)


        # # Spark ML pipeline
        # simple_pipline_builder = LGBAdvancedPipelineTmp(**ml_alg_kwargs)
        # spark_pipeline = simple_pipline_builder.create_pipeline(sdataset)
        # spark_pipeline_model = spark_pipeline.fit(sdataset.data)
        # spark_pipeline_model.transform(sdataset.data).toPandas().to_csv("/tmp/spark_pipeline_model.csv", index=False)

        logger.info("Finished")

