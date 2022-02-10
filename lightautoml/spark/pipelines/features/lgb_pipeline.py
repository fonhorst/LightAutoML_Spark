from copy import deepcopy
from typing import cast, Optional, Union, List, Set
from unicodedata import name

import numpy as np

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import FeaturesPipeline, FeaturesPipelineSpark, TabularDataFeaturesSpark
from lightautoml.spark.pipelines.features.base import TabularDataFeatures
from lightautoml.spark.transformers.base import ChangeRolesTransformer, SparkTransformer, SequentialTransformer, \
    UnionTransformer, \
    ColumnsSelector, ChangeRoles, SparkUnionTransformer, SparkSequentialTransformer
from lightautoml.spark.transformers.categorical import OrdinalEncoder, TargetEncoderEstimator
from lightautoml.spark.transformers.datetime import TimeToNum, SparkTimeToNumTransformer

from pyspark.ml import Transformer, Pipeline, Estimator
from lightautoml.spark.transformers.categorical import OrdinalEncoderEstimator
from lightautoml.spark.transformers.base import ColumnsSelectorTransformer


class LGBSimpleFeaturesSpark(FeaturesPipelineSpark, TabularDataFeaturesSpark):
    """Creates simple pipeline for tree based models.

    Simple but is ok for select features.
    Numeric stay as is, Datetime transforms to numeric.
    Categorical label encoding.
    Maps input to output features exactly one-to-one.

    """
    def __init__(self, input_features: List[str], input_roles: RolesDict):
        super().__init__()
        self._input_features = input_features
        self._input_roles = input_roles

    def _get_input_features(self) -> Set[str]:
        return set(self.input_features)

    def create_pipeline(self, train: SparkDataset) -> Union[SparkUnionTransformer, SparkSequentialTransformer]:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        # TODO: Transformer params to config
        transformers_list = []

        # process categories
        categories = self._cols_by_role(train, "Category")
        if len(categories) > 0:
            roles = {f: train.roles[f] for f in categories}
            cat_processing = OrdinalEncoderEstimator(input_cols=categories,
                                                     input_roles=roles,
                                                     subs=None,
                                                     random_state=42)
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = self._cols_by_role(train, "Datetime")
        if len(datetimes) > 0:
            roles = {f: train.roles[f] for f in datetimes}
            dt_processing = SparkTimeToNumTransformer(input_cols=datetimes,
                                                      input_roles=roles)
            transformers_list.append(dt_processing)

        union_all = SparkUnionTransformer(transformers_list)

        return union_all


class LGBSimpleFeatures(FeaturesPipeline):
    """Creates simple pipeline for tree based models.

    Simple but is ok for select features.
    Numeric stay as is, Datetime transforms to numeric.
    Categorical label encoding.
    Maps input to output features exactly one-to-one.

    """

    def create_pipeline(self, train: SparkDataset) -> SparkTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        # TODO: Transformer params to config
        transformers_list = []

        # process categories
        categories = get_columns_by_role(train, "Category")
        if len(categories) > 0:
            cat_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=categories),
                    OrdinalEncoder(subs=None, random_state=42),
                    # ChangeRoles(NumericRole(np.float32))
                ]
            )
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = get_columns_by_role(train, "Datetime")
        if len(datetimes) > 0:
            dt_processing = SequentialTransformer([ColumnsSelector(keys=datetimes), TimeToNum()])
            transformers_list.append(dt_processing)

        # process numbers
        numerics = get_columns_by_role(train, "Numeric")
        if len(numerics) > 0:
            num_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=numerics),
                    # ConvertDataset(dataset_type=NumpyDataset),
                ]
            )
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all

    def _merge(self, data: SparkDataset) -> SparkTransformer:
        pipes = [cast(SparkTransformer, pipe(data))
                 for pipe in self.pipes]

        union = UnionTransformer(pipes) if len(pipes) > 1 else pipes[0]
        print(f"Producing union: {type(union)}")
        return union

    def _merge_seq(self, data: SparkDataset) -> SparkTransformer:
        pipes = [cast(SparkTransformer, pipe(data))
                 for pipe in self.pipes]
        # pipes = []
        # for pipe in self.pipes:
        #     _pipe = pipe(data)
        #     data = _pipe.fit_transform(data)
        #     pipes.append(_pipe)

        seq = SequentialTransformer(pipes) if len(pipes) > 1 else pipes[0]
        return seq


class LGBSimpleFeaturesTmp(FeaturesPipeline):

    def create_pipeline(self, train: SparkDataset) -> Pipeline:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """

        final_columns = []
        stages = []

        categories = get_columns_by_role(train, "Category")
        if len(categories) > 0:
            ord_estimator = OrdinalEncoderEstimator(input_cols=categories, input_roles=train.roles)
            stages.append(ord_estimator)
            final_columns = categories + ord_estimator.getOutputCols()

        datetimes = get_columns_by_role(train, "Datetime")
        if len(datetimes) > 0:
            final_columns = final_columns + datetimes

        numerics = get_columns_by_role(train, "Numeric")
        if len(numerics) > 0:
            final_columns = final_columns + numerics

        columns_selector = ColumnsSelectorTransformer(input_cols=final_columns)
        stages.append(columns_selector)

        pipeline = Pipeline(stages=stages)

        return pipeline


# we don't inherit from LAMALGBAdvancedPipeline
# because his method 'create_pipeline' contains direct
# calls to Sequential and Union transformers
class LGBAdvancedPipeline(FeaturesPipeline, TabularDataFeatures):
    """Create advanced pipeline for trees based models.

        Includes:

            - Different cats and numbers handling according to role params.
            - Dates handling - extracting seasons and create datediffs.
            - Create categorical intersections.

        """

    def __init__(
            self,
            feats_imp: Optional[ImportanceEstimator] = None,
            top_intersections: int = 5,
            max_intersection_depth: int = 3,
            subsample: Optional[Union[int, float]] = None,
            multiclass_te_co: int = 3,
            auto_unique_co: int = 10,
            output_categories: bool = False,
            **kwargs
    ):
        """

        Args:
            feats_imp: Features importances mapping.
            top_intersections: Max number of categories
              to generate intersections.
            max_intersection_depth: Max depth of cat intersection.
            subsample: Subsample to calc data statistics.
            multiclass_te_co: Cutoff if use target encoding in cat
              handling on multiclass task if number of classes is high.
            auto_unique_co: Switch to target encoding if high cardinality.

        """
        print("lama advanced pipeline ctr")
        super().__init__(
            multiclass_te_co=multiclass_te_co,
            top_intersections=top_intersections,
            max_intersection_depth=max_intersection_depth,
            subsample=subsample,
            feats_imp=feats_imp,
            auto_unique_co=auto_unique_co,
            output_categories=output_categories,
            ascending_by_cardinality=False,
        )

    def create_pipeline(self, train: SparkDataset) -> SparkTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """

        transformer_list = []
        target_encoder = self.get_target_encoder(train)

        output_category_role = (
            CategoryRole(np.float32, label_encoded=True) if self.output_categories else NumericRole(np.float32)
        )

        # handle categorical feats
        # split categories by handling type. This pipe use 3 encodings - freq/label/target/ordinal
        # 1 - separate freqs. It does not need label encoding
        transformer_list.append(self.get_freq_encoding(train))

        # 2 - check different target encoding parts and split (ohe is the same as auto - no ohe in gbm)
        auto = get_columns_by_role(train, "Category", encoding_type="auto") + get_columns_by_role(
            train, "Category", encoding_type="ohe"
        )

        if self.output_categories:
            le = (
                    auto
                    + get_columns_by_role(train, "Category", encoding_type="oof")
                    + get_columns_by_role(train, "Category", encoding_type="int")
            )
            te = []
            ordinal = None

        else:
            le = get_columns_by_role(train, "Category", encoding_type="int")
            ordinal = get_columns_by_role(train, "Category", ordinal=True)

            if target_encoder is not None:
                te = get_columns_by_role(train, "Category", encoding_type="oof")
                # split auto categories by unique values cnt
                un_values = self.get_uniques_cnt(train, auto)
                te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
                ordinal = ordinal + list(set(auto) - set(te))

            else:
                te = []
                ordinal = ordinal + auto + get_columns_by_role(train, "Category", encoding_type="oof")

            ordinal = sorted(list(set(ordinal)))

        # get label encoded categories
        le_part = self.get_categorical_raw(train, le)
        if le_part is not None:
            le_part = SequentialTransformer([le_part, ChangeRoles(output_category_role)])
            transformer_list.append(le_part)

        # get target encoded part
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            te_part = SequentialTransformer([te_part, target_encoder()])
            transformer_list.append(te_part)

        # TODO: SPARK-LAMA fix bug with performance of catintersections
        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                ints_part = SequentialTransformer([intersections, target_encoder()])
            else:
                ints_part = SequentialTransformer([intersections, ChangeRoles(output_category_role)])

            transformer_list.append(ints_part)

        # add numeric pipeline
        transformer_list.append(self.get_numeric_data(train))
        transformer_list.append(self.get_ordinal_encoding(train, ordinal))
        # add difference with base date
        transformer_list.append(self.get_datetime_diffs(train))
        # add datetime seasonality
        transformer_list.append(self.get_datetime_seasons(train, NumericRole(np.float32)))

        # final pipeline
        union_all = UnionTransformer([x for x in transformer_list if x is not None])

        return union_all

    def _merge(self, data: SparkDataset) -> SparkTransformer:
        pipes = [cast(SparkTransformer, pipe(data))
                 for pipe in self.pipes]

        union = UnionTransformer(pipes) if len(pipes) > 1 else pipes[0]
        print(f"Producing union: {type(union)}")
        return union

    def _merge_seq(self, data: SparkDataset) -> SparkTransformer:
        pipes = [cast(SparkTransformer, pipe(data))
                 for pipe in self.pipes]
        # pipes = []
        # for pipe in self.pipes:
        #     _pipe = pipe(data)
        #     data = _pipe.fit_transform(data)
        #     pipes.append(_pipe)

        seq = SequentialTransformer(pipes) if len(pipes) > 1 else pipes[0]
        return seq

class LGBAdvancedPipelineTmp(FeaturesPipeline, TabularDataFeatures):

    def create_pipeline(self, train: SparkDataset) -> Pipeline:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """

        features = train.features
        roles = deepcopy(train.roles)
        transformer_list = []
        
        target_encoder = self.get_target_encoder_new(train)

        output_category_role = (
            CategoryRole(np.float32, label_encoded=True) if self.output_categories else NumericRole(np.float32)
        )

        # handle categorical feats
        # split categories by handling type. This pipe use 3 encodings - freq/label/target/ordinal
        # 1 - separate freqs. It does not need label encoding
        stage = self.get_freq_encoding_new(train)
        if stage:
            transformer_list.append(stage)
            features = features + stage.getOutputCols()
            roles.update(stage.getOutputRoles())

        # 2 - check different target encoding parts and split (ohe is the same as auto - no ohe in gbm)
        auto = get_columns_by_role(train, "Category", encoding_type="auto") + get_columns_by_role(
            train, "Category", encoding_type="ohe"
        )

        if self.output_categories:
            le = (
                    auto
                    + get_columns_by_role(train, "Category", encoding_type="oof")
                    + get_columns_by_role(train, "Category", encoding_type="int")
            )
            te = []
            ordinal = None

        else:
            le = get_columns_by_role(train, "Category", encoding_type="int")
            ordinal = get_columns_by_role(train, "Category", ordinal=True)

            if target_encoder is not None:
                te = get_columns_by_role(train, "Category", encoding_type="oof")
                # split auto categories by unique values cnt
                un_values = self.get_uniques_cnt(train, auto)
                te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
                ordinal = ordinal + list(set(auto) - set(te))

            else:
                te = []
                ordinal = ordinal + auto + get_columns_by_role(train, "Category", encoding_type="oof")

            ordinal = sorted(list(set(ordinal)))

        # get label encoded categories
        le_part = self.get_categorical_raw_new(train, le)
        if le_part is not None:
            # le_part = SequentialTransformer([le_part, ChangeRoles(output_category_role)])
            change_roles_stage = ChangeRolesTransformer(input_cols=le_part.getOutputCols(),
                                                        roles=output_category_role)
            features = features + le_part.getOutputCols()
            roles.update(change_roles_stage.getOutputRoles())
            le_part = SequentialTransformer([le_part, change_roles_stage])
            transformer_list.append(le_part)

        # get target encoded part
        te_part = self.get_categorical_raw_new(train, te)
        if te_part is not None:
            # te_part = SequentialTransformer([te_part, target_encoder()])
            target_encoder_stage = target_encoder(input_cols=te_part.getOutputCols(),
                                         input_roles=te_part.getOutputRoles(),
                                         task_name=train.task.name,
                                         folds_column=train.folds_column,
                                         target_column=train.target_column)
            features = features + target_encoder_stage.getOutputCols()
            roles.update(target_encoder_stage.getOutputRoles())                                         
            te_part = SequentialTransformer([te_part, target_encoder_stage])
            transformer_list.append(te_part)


        # TODO: SPARK-LAMA fix bug with performance of catintersections
        # get intersection of top categories
        intersections = self.get_categorical_intersections_new(train)
        if intersections is not None:
            if target_encoder is not None:
                # ints_part = SequentialTransformer([intersections, target_encoder()])
                target_encoder_stage = target_encoder(input_cols=intersections.getOutputCols(),
                                             input_roles=intersections.getOutputRoles(),
                                             task_name=train.task.name,
                                             folds_column=train.folds_column,
                                             target_column=train.target_column)
                features = features + target_encoder_stage.getOutputCols()
                roles.update(target_encoder_stage.getOutputRoles())
                ints_part = SequentialTransformer([intersections, target_encoder_stage])
                transformer_list.append(ints_part)

            else:
                # ints_part = SequentialTransformer([intersections, ChangeRoles(output_category_role)])
                change_roles_stage = ChangeRolesTransformer(input_cols=intersections.getOutputCols(),
                                                           role=output_category_role)
                features = features + intersections.getOutputCols()
                roles.update(change_roles_stage.getOutputRoles())
                ints_part = SequentialTransformer([intersections, change_roles_stage])
                transformer_list.append(ints_part)

        # add numeric pipeline
        if stage := self.get_numeric_data_new(train):
            transformer_list.append(stage)
            roles.update(stage.getOutputRoles())
        if stage := self.get_ordinal_encoding_new(train, ordinal):
            transformer_list.append(stage)
            features = features + stage.getOutputCols()
            roles.update(stage.getOutputRoles())
        # add difference with base date
        if stage := self.get_datetime_diffs_new(train):
            transformer_list.append(stage)
            features = features + stage.getOutputCols()
            roles.update(stage.getOutputRoles())
        # add datetime seasonality
        if stage := self.get_datetime_seasons_new(train, NumericRole(np.float32)):
            transformer_list.append(stage)
            features = features + stage.getOutputCols()
            roles.update(stage.getOutputRoles())

        # final pipeline
        union_all = UnionTransformer([x for x in transformer_list if x is not None])

        return union_all, features, roles
