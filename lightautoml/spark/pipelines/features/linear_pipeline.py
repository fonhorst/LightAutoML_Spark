from typing import List, Optional, Set, Union, cast
import numpy as np

from lightautoml.dataset.base import LAMLDataset, RolesDict
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures as LAMALinearFeatures
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import SparkFeaturesPipeline, SparkTabularDataFeatures

from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.dataset.roles import CategoryRole

# Same comments as for spark.pipelines.features.base
from lightautoml.spark.transformers.categorical import SparkOHEEncoderEstimator, SparkLabelEncoderEstimator
from lightautoml.spark.transformers.numeric import FillInf, SparkFillInfTransformer, FillnaMedian, SparkFillnaMedianEstimator, \
    LogOdds, SparkLogOddsTransformer, SparkNaNFlagsEstimator, StandardScaler, NaNFlags, SparkStandardScalerEstimator

from lightautoml.spark.transformers.base import ChangeRolesTransformer, SparkUnionTransformer, SparkSequentialTransformer, SparkEstOrTrans


class SparkLinearFeatures(SparkFeaturesPipeline, SparkTabularDataFeatures):
    def __init__(
        self,
        feats_imp: Optional[ImportanceEstimator] = None,
        top_intersections: int = 5,
        max_bin_count: int = 10,
        max_intersection_depth: int = 3,
        subsample: Optional[Union[int, float]] = None,
        sparse_ohe: Union[str, bool] = "auto",
        auto_unique_co: int = 50,
        output_categories: bool = True,
        multiclass_te_co: int = 3,
        **_
    ):
        """

        Args:
            feats_imp: Features importances mapping.
            top_intersections: Max number of categories
              to generate intersections.
            max_bin_count: Max number of bins to discretize numbers.
            max_intersection_depth: Max depth of cat intersection.
            subsample: Subsample to calc data statistics.
            sparse_ohe: Should we output sparse if ohe encoding
              was used during cat handling.
            auto_unique_co: Switch to target encoding if high cardinality.
            output_categories: Output encoded categories or embed idxs.
            multiclass_te_co: Cutoff if use target encoding in cat handling
              on multiclass task if number of classes is high.

        """
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
        # self._input_features = input_features
        # self._input_roles = input_roles

    def _get_input_features(self) -> Set[str]:
        return set(self.input_features)

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """Create linear pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """
        transformers_list = []
        dense_list = []
        sparse_list = []
        probs_list = []
        target_encoder = self.get_target_encoder(train)
        te_list = dense_list if train.task.name == "reg" else probs_list

        # handle categorical feats
        # split categories by handling type. This pipe use 4 encodings - freq/label/target/ohe/ordinal
        # 1 - separate freqs. It does not need label encoding
        dense_list.append(self.get_freq_encoding(train))

        # 2 - check 'auto' type (int is the same - no label encoded numbers in linear models)
        auto = self._cols_by_role(train, "Category", encoding_type="auto") + self._cols_by_role(
            train, "Category", encoding_type="int"
        )

        # if self.output_categories or target_encoder is None:
        if target_encoder is None:
            le = (
                    auto
                    + self._cols_by_role(train, "Category", encoding_type="oof")
                    + self._cols_by_role(train, "Category", encoding_type="ohe")
            )
            te = []

        else:
            te = self._cols_by_role(train, "Category", encoding_type="oof")
            le = self._cols_by_role(train, "Category", encoding_type="ohe")
            # split auto categories by unique values cnt
            un_values = self.get_uniques_cnt(train, auto)
            te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
            le = le + list(set(auto) - set(te))

        # get label encoded categories
        sparse_list.append(self.get_categorical_raw(train, le))

        # TODO: fix the performance and uncomment
        # get target encoded categories
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            target_encoder_stage = target_encoder(
                input_cols=te_part.getOutputCols(),
                input_roles=te_part.getOutputRoles(),
                task_name=train.task.name,
                folds_column=train.folds_column,
                target_column=train.target_column,
                do_replace_columns=True)
            te_part = SparkSequentialTransformer([te_part, target_encoder_stage])
            te_list.append(te_part)
        
        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                target_encoder_stage = target_encoder(
                    input_cols=intersections.getOutputCols(),
                    input_roles=intersections.getOutputRoles(),
                    task_name=train.task.name,
                    folds_column=train.folds_column,
                    target_column=train.target_column,
                    do_replace_columns=True
                )
                ints_part = SparkSequentialTransformer([intersections, target_encoder_stage])
                te_list.append(ints_part)
            else:
                sparse_list.append(intersections)

        # add datetime seasonality
        seas_cats = self.get_datetime_seasons(train, CategoryRole(np.int32))
        if seas_cats is not None:
            # sparse_list.append(SequentialTransformer([seas_cats, LabelEncoder()]))
            label_encoder_stage = SparkLabelEncoderEstimator(input_cols=seas_cats.outputCols(),
                                                             input_roles=seas_cats.outputRoles(),
                                                             do_replace_columns=True)
            sparse_list.append(SparkSequentialTransformer([seas_cats, label_encoder_stage]))

        # get quantile binning
        sparse_list.append(self.get_binned_data(train))
        # add numeric pipeline wo probs
        dense_list.append(self.get_numeric_data(train, prob=False))
        # add ordinal categories
        dense_list.append(self.get_ordinal_encoding(train))
        # add probs
        probs_list.append(self.get_numeric_data(train, prob=True))
        # add difference with base date
        dense_list.append(self.get_datetime_diffs(train))

        # combine it all together
        # handle probs if exists
        probs_list = [x for x in probs_list if x is not None]
        if len(probs_list) > 0:
            probs_pipe = SparkUnionTransformer(probs_list)
            probs_pipe = SparkSequentialTransformer([probs_pipe, SparkLogOddsTransformer(input_cols=probs_pipe.get_output_cols(),
                                                                                         input_roles=probs_pipe.get_output_roles())])
            dense_list.append(probs_pipe)

        # handle dense
        dense_list = [x for x in dense_list if x is not None]
        if len(dense_list) > 0:
            # standartize, fillna, add null flags

            dense_pipe1 = SparkUnionTransformer(dense_list)
            fill_inf_stage = SparkFillInfTransformer(input_cols=dense_pipe1.get_output_cols(),
                                                     input_roles=dense_pipe1.get_output_roles())
            fill_na_median_stage = SparkFillnaMedianEstimator(input_cols=fill_inf_stage.getOutputCols(),
                                                              input_roles=fill_inf_stage.getOutputCols())
            standerd_scaler_stage = SparkStandardScalerEstimator(input_cols=fill_na_median_stage.getOutputCols(),
                                                                 input_roles=fill_na_median_stage.getOutputRoles())

            dense_pipe = SparkSequentialTransformer(
                [
                    dense_pipe1,
                    SparkUnionTransformer(
                        [
                            SparkSequentialTransformer([fill_inf_stage, fill_na_median_stage, standerd_scaler_stage]),
                            SparkNaNFlagsEstimator(input_cols=dense_pipe1.get_output_cols(),
                                                   input_roles=dense_pipe1.get_output_roles()),
                        ]
                    ),
                ]
            )
            transformers_list.append(dense_pipe)

        # handle categories - cast to float32 if categories are inputs or make ohe
        sparse_list = [x for x in sparse_list if x is not None]
        if len(sparse_list) > 0:
            sparse_pipe = SparkUnionTransformer(sparse_list)
            if self.output_categories:
                final = ChangeRolesTransformer(input_cols=sparse_pipe.get_output_cols(),
                                               input_roles=sparse_pipe.get_output_roles(),
                                               role=CategoryRole(np.float32))
            else:
                if self.sparse_ohe == "auto":
                    final = SparkOHEEncoderEstimator(input_cols=sparse_pipe.get_output_cols(),
                                                     input_roles=sparse_pipe.get_output_roles(),
                                                     total_feats_cnt=train.shape[1])
                else:
                    final = SparkOHEEncoderEstimator(input_cols=sparse_pipe.get_output_cols(),
                                                     input_roles=sparse_pipe.get_output_roles(),
                                                     make_sparse=self.sparse_ohe)
            sparse_pipe = SparkSequentialTransformer([sparse_pipe, final])

            transformers_list.append(sparse_pipe)

        # final pipeline
        union_all = SparkUnionTransformer(transformers_list)

        return union_all