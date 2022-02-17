import logging
from copy import deepcopy
from typing import Optional, Any, List, Dict, Tuple

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, NumericType, DoubleType, FloatType

from lightautoml.dataset.base import array_attr_roles, valid_array_attributes
from lightautoml.dataset.roles import ColumnRole, DropRole, NumericRole, DatetimeRole, CategoryRole
from lightautoml.dataset.utils import roles_parser
from lightautoml.reader.base import Reader, UserDefinedRolesDict, RoleType, RolesDict
from lightautoml.reader.guess_roles import calc_encoding_rules, rule_based_roles_guess, calc_category_rules, \
    rule_based_cat_handler_guess
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.reader.guess_roles import get_numeric_roles_stat, get_category_roles_stat, get_null_scores
from lightautoml.tasks import Task
from lightautoml.utils.tmp_utils import log_data

logger = logging.getLogger(__name__)

dtype2Stype ={
    "str": "string",
    "bool": "boolean",
    "int": "int",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "int128": "bigint",
    "int256": "bigint",
    "integer": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "uint128": "bigint",
    "uint256": "bigint",
    "longlong": "long",
    "ulonglong": "long",
    "float16": "float",
    "float": "float",
    "float32": "float",
    "float64": "double",
    "float128": "double"
}

stype2dtype = {
    "string": "str",
    "boolean": "bool",
    "bool": "bool",
    "int": "int",
    "bigint": "longlong",
    "long": "long",
    "float": "float",
    "double": "float64"
}


class SparkToSparkReader(Reader):
    """
    Reader to convert :class:`~pandas.DataFrame` to AutoML's :class:`~lightautoml.dataset.np_pd_dataset.PandasDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(
        self,
        task: Task,
        samples: Optional[int] = 100000,
        max_nan_rate: float = 0.999,
        max_constant_rate: float = 0.999,
        cv: int = 5,
        random_state: int = 42,
        roles_params: Optional[dict] = None,
        n_jobs: int = 4,
        # params for advanced roles guess
        advanced_roles: bool = True,
        numeric_unique_rate: float = 0.999,
        max_to_3rd_rate: float = 1.1,
        binning_enc_rate: float = 2,
        raw_decr_rate: float = 1.1,
        max_score_rate: float = 0.2,
        abs_score_val: float = 0.04,
        drop_score_co: float = 0.01,
        **kwargs: Any
    ):
        """

        Args:
            task: Task object.
            samples: Number of elements used when checking role type.
            max_nan_rate: Maximum nan-rate.
            max_constant_rate: Maximum constant rate.
            cv: CV Folds.
            random_state: Random seed.
            roles_params: dict of params of features roles. \
                Ex. {'numeric': {'dtype': np.float32}, 'datetime': {'date_format': '%Y-%m-%d'}}
                It's optional and commonly comes from config
            n_jobs: Int number of processes.
            advanced_roles: Param of roles guess (experimental, do not change).
            numeric_unqiue_rate: Param of roles guess (experimental, do not change).
            max_to_3rd_rate: Param of roles guess (experimental, do not change).
            binning_enc_rate: Param of roles guess (experimental, do not change).
            raw_decr_rate: Param of roles guess (experimental, do not change).
            max_score_rate: Param of roles guess (experimental, do not change).
            abs_score_val: Param of roles guess (experimental, do not change).
            drop_score_co: Param of roles guess (experimental, do not change).
            **kwargs: For now not used.

        """
        super().__init__(task)
        self.samples = samples
        self.max_nan_rate = max_nan_rate
        self.max_constant_rate = max_constant_rate
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.roles_params = roles_params
        self.target = None
        self.target_col = None
        if roles_params is None:
            self.roles_params = {}

        self.advanced_roles = advanced_roles
        self.advanced_roles_params = {
            "numeric_unique_rate": numeric_unique_rate,
            "max_to_3rd_rate": max_to_3rd_rate,
            "binning_enc_rate": binning_enc_rate,
            "raw_decr_rate": raw_decr_rate,
            "max_score_rate": max_score_rate,
            "abs_score_val": abs_score_val,
            "drop_score_co": drop_score_co,
        }

        self.params = kwargs

    def fit_read(
        self, train_data: SparkDataFrame, features_names: Any = None, roles: UserDefinedRolesDict = None, **kwargs: Any
    ) -> SparkDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        logger.info(f"\x1b[1mTrain data columns: {train_data.columns}\x1b[0m\n")

        if SparkDataset.ID_COLUMN not in train_data.columns:
            train_data = train_data.withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
        train_data = train_data.cache()

        if roles is None:
            roles = {}

        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole
        attrs_dict = dict(zip(array_attr_roles, valid_array_attributes))

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if type(r) == str:
                # get default role params if defined
                r = self._get_default_role_from_str(r)

            # check if column is defined like target/group/weight etc ...
            if r.name in attrs_dict:
                # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?
                # TODO: Think, what if multilabel or multitask? Multiple column target ..
                # TODO: Maybe for multilabel/multitask make target only avaliable in kwargs??
                self._used_array_attrs[attrs_dict[r.name]] = feat
                kwargs[attrs_dict[r.name]] = feat
                r = DropRole()

            # add new role
            parsed_roles[feat] = r

        if "target" in kwargs:
            assert isinstance(kwargs["target"], str), \
                f"Target must be a column or column name, but it is {type(kwargs['target'])}"
            assert kwargs["target"] in train_data.columns, \
                f"Target must be a part of dataframe. Target: {kwargs['target']}"
            # self.target = kwargs["target"]
            self.target_col = kwargs["target"]
        elif "target" in train_data.columns:
            # self.target = "target"
            self.target_col = "target"
        elif "TARGET" in train_data.columns:
            # self.target = "TARGET"
            self.target_col = "TARGET"
        else:
            assert False,  "Target should be defined"
        # assert "target" in kwargs or "target" in [c.lower() for c in train_data.columns], "Target should be defined"
        # self.target = kwargs["target"] if "target" in kwargs else "target"

        train_data = self._create_target(train_data, target_col=self.target_col)

        self.target = train_data.select(SparkDataset.ID_COLUMN, self.target_col)
        train_data = train_data.select(*[c for c in train_data.columns if c != self.target_col])

        # get subsample if it needed
        subsample = train_data
        # TODO: LAMA-SPARK replace seed here with a variable
        if self.samples:
            subsample = subsample.sample(fraction=0.1, seed=42).limit(self.samples).cache()

        # TODO: LAMA-SPARK rewrite this part:
        #   Implement the checking logic
        #   1. get spark df schema
        #   2. get feat in parsed roles
        #   3. parsed role matches Spark datatype ?
        #       if yes, just accept
        #       if no, try to test and convert
        #   4. there is no parsed role, try to match from Spark schema (datetime, ints, doubles)
        #   5. if not possible (e.g. string type), try to guess using LAMA logic

        # infer roles
        feats_to_guess: List[str] = []
        inferred_feats: Dict[str, ColumnRole] = dict()
        for feat in subsample.columns:
            if feat == SparkDataset.ID_COLUMN:
                continue

            assert isinstance(
                feat, str
            ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))
            if feat in parsed_roles:
                r = parsed_roles[feat]
                # handle datetimes
                if r.name == "Datetime":
                    # try if it's ok to infer date with given params
                    result = subsample.select(
                        F.to_timestamp(feat, format=r.format).isNotNull().astype(IntegerType()).alias(f"{feat}_dt"),
                        F.count('*')
                    ).first()

                    if result[f"{feat}_dt"] != result['count']:
                        raise ValueError("Looks like given datetime parsing params are not correctly defined")

                # replace default category dtype for numeric roles dtype if cat col dtype is numeric
                if r.name == "Category":
                    # default category role
                    cat_role = self._get_default_role_from_str("category")
                    # check if role with dtypes was exactly defined
                    flg_default_params = feat in roles["category"] if "category" in roles else False

                    inferred_dtype = next(dtyp for fname, dtyp in subsample.dtypes if fname == feat)
                    inferred_dtype = np.dtype(inferred_dtype)

                    if (
                        flg_default_params
                        and not np.issubdtype(cat_role.dtype, np.number)
                        and np.issubdtype(inferred_dtype, np.number)
                    ):
                        r.dtype = self._get_default_role_from_str("numeric").dtype

                inferred_feats[feat] = r
            else:
                # TODO: SPARK-LAMA this functions can be applied for all columns in a single round
                feats_to_guess.append(feat)
                # # if no - infer
                # if self._is_ok_feature(subsample, feat):
                #     r = self._guess_role(subsample, feat)
                # else:
                #     r = DropRole()

        ok_features = self._ok_features(train_data, feats_to_guess)
        guessed_feats = self._guess_role(train_data, ok_features)
        inferred_feats.update(guessed_feats)

        # # set back
        for feat, r in inferred_feats.items():
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(self.used_features) > 0, "All features are excluded for some reasons"
        # assert len(self.used_array_attrs) > 0, 'At least target should be defined in train dataset'
        # create folds

        kwargs["folds"] = self._create_folds(train_data, kwargs)
        # get dataset
        # replace data target with freshly created spark dataframe
        kwargs["target"] = self.target

        # TODO: SPARK-LAMA send parent for unwinding
        ff = [
            F.when(F.isnull(f), float('nan')).otherwise(F.col(f).astype(FloatType())).alias(f) if isinstance(self.roles[f], NumericRole) else f
            for f in self.used_features
        ]

        train_data = (
            train_data
            .select(SparkDataset.ID_COLUMN, *ff)
        )

        dataset = SparkDataset(
            train_data,
            self.roles,
            task=self.task,
            **kwargs
        )

        # TODO: SPARK-LAMA will be implemented later
        if self.advanced_roles:
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)
        
            droplist = [x for x in new_roles if new_roles[x].name == "Drop" and not self._roles[x].force_input]
            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
        
            # TODO: SPARK-LAMA send parent for unwinding
            dataset = SparkDataset(
                train_data.select(SparkDataset.ID_COLUMN, *self.used_features),
                self.roles,
                task=self.task,
                **kwargs
            )

        # ds = dataset.to_pandas()
        # ds.task = None
        log_data("spark_reader_fit_read", {"train": dataset.to_pandas()})

        return dataset

    def _create_folds(self, sdf: SparkDataFrame, kwargs: dict) -> Optional[SparkDataFrame]:
        if "folds" in kwargs:
            folds_col = kwargs["folds"]

            assert isinstance(folds_col, str), \
                f"If kwargs contains 'folds' it should be of type str and contain folds column." \
                f"But kwargs['folds'] has type {type(folds_col)} and contains {folds_col}"

            assert folds_col in sdf.columns, \
                f"Folds column ({folds_col}) should be presented in the train dataframe," \
                f"but it is not possible to find the column among {sdf.columns}"

            cols_dtypes = dict(sdf.dtypes)
            assert cols_dtypes[folds_col] == 'int', \
                f"Folds column should be of integer type, but it is {cols_dtypes[folds_col]}"

            folds_sdf = sdf.select(SparkDataset.ID_COLUMN, folds_col)
            return folds_sdf

            # # TODO: need to validate the column
            # df = sdf
            # datasets = []
            # for i in range(self.cv):
            #     condition = sdf[folds_col] == i
            #     # filtering and dropping folds col
            #     validation = df.filter(condition).drop(folds_col)
            #     train = df.filter(~condition).drop(folds_col)
            #     datasets.append((train, validation))
            # return datasets

        if self.cv == 1:
            return None

        h = 1.0 / self.cv
        folds_sdf = sdf.select(
            SparkDataset.ID_COLUMN,
            F.floor(F.rand(self.random_state) / h).alias("reader_fold_num")
        )
        return folds_sdf
        # h = 1.0 / self.cv
        # rands_col = "fold_random_num"
        # df = sdf.select('*', F.rand(self.random_state).alias(rands_col))
        # datasets = []
        # for i in range(self.cv):
        #     validateLB = i * h
        #     validateUB = (i + 1) * h
        #     condition = (df[rands_col] >= validateLB) & (df[rands_col] < validateUB)
        #     # filtering and dropping rand num col
        #     validation = df.filter(condition).drop(rands_col)
        #     train = df.filter(~condition).drop(rands_col)
        #     datasets.append((train, validation))
        #
        # return datasets

    def _create_target(self, sdf: SparkDataFrame, target_col: str = "target"):
        """Validate target column and create class mapping is needed

        Args:
            target: Column with target values.

        Returns:
            Transformed target.

        """
        self.class_mapping = None

        nan_count = sdf.where(F.isnan(target_col)).count()
        assert nan_count == 0, "Nan in target detected"

        if self.task.name != "reg":
            uniques = sdf.select(target_col).distinct().collect()
            uniques = [r[target_col] for r in uniques]
            self._n_classes = len(uniques)
            # dict(sdf.dtypes)[target_col]

            # TODO: make it pretty
            # 1. check the column type if it is numeric or not
            # 2. if it is string can i convert it to num?
            if isinstance(sdf.schema[target_col], NumericType):
                srtd = np.ndarray(sorted(uniques))
            # elif isinstance(sdf.schema[target_col], StringType):
            #     try:
            #         srtd = np.ndarray(sorted([int(el) for el in uniques]))
            #         sdf = sdf
            #     except ValueError:
            #         srtd = None
            else:
                srtd = None

            if srtd and (np.arange(srtd.shape[0]) == srtd).all():

                assert srtd.shape[0] > 1, "Less than 2 unique values in target"
                if self.task.name == "binary":
                    assert srtd.shape[0] == 2, "Binary task and more than 2 values in target"
                return sdf

            mapping = {x: i for i, x in enumerate(uniques)}

            remap = F.udf(lambda x: mapping[x], returnType=IntegerType())
            rest_cols = [c for c in sdf.columns if c != target_col]
            sdf_with_proc_target = sdf.select(*rest_cols, remap(target_col).alias(target_col))

            self.class_mapping = mapping

            return sdf_with_proc_target
        else:
            rest_cols = [c for c in sdf.columns if c != target_col]
            sdf = sdf.select(*rest_cols, F.col(target_col).astype(DoubleType()).alias(target_col))

        return sdf

    def _get_default_role_from_str(self, name) -> RoleType:
        """Get default role for string name according to automl's defaults and user settings.

        Args:
            name: name of role to get.

        Returns:
            role object.

        """
        name = name.lower()
        try:
            role_params = self.roles_params[name]
        except KeyError:
            role_params = {}

        return ColumnRole.from_string(name, **role_params)

    def _guess_role(self, data: SparkDataFrame, features: List[Tuple[str, bool]]) -> Dict[str, RoleType]:
        """Try to infer role, simple way.

        If convertable to float -> number.
        Else if convertable to datetime -> datetime.
        Else category.

        Args:
            feature: Column from dataset.

        Returns:
            Feature role.

        """
        guessed_cols = dict()
        cols_to_check = []
        check_columns = []

        feat2dtype = dict(data.dtypes)

        for feature, ok in features:
            if not ok:
                guessed_cols[feature] = DropRole()
                continue
            inferred_dtype = feat2dtype[feature]
            # numpy doesn't understand 'string' but 'str' is ok
            inferred_dtype = np.dtype(stype2dtype[inferred_dtype])

            # testing if it can be numeric or not
            num_dtype = self._get_default_role_from_str("numeric").dtype
            date_format = self._get_default_role_from_str("datetime").format
            # TODO: can it be really converted?
            if np.issubdtype(inferred_dtype, bool):
                guessed_cols[feature] = CategoryRole(dtype=np.bool8)
                continue
            elif np.issubdtype(inferred_dtype, np.number):
                guessed_cols[feature] = NumericRole(num_dtype)
                continue

            fcol = F.col(feature)

            can_cast_to_numeric = (
                F.when(F.isnull(fcol), True)
                .otherwise(fcol.cast(dtype2Stype[num_dtype.__name__]).isNotNull())
                .astype(IntegerType())
            )

            # TODO: utc handling here?
            can_cast_to_datetime = F.to_timestamp(feature, format=date_format).isNotNull().astype(IntegerType())

            cols_to_check.append((feature, num_dtype, date_format))
            check_columns.extend([
                F.sum(can_cast_to_numeric).alias(f"{feature}_num"),
                F.sum(can_cast_to_datetime).alias(f"{feature}_dt"),
            ])

        result = data.select(
            *check_columns,
            F.count('*').alias('count')
        ).first()

        for feature, num_dtype, date_format in cols_to_check:
            if result[f"{feature}_num"] == result['count']:
                guessed_cols[feature] = NumericRole(num_dtype)
            elif result[f"{feature}_dt"] == result['count']:
                guessed_cols[feature] = DatetimeRole(np.datetime64, date_format=date_format)
            else:
                guessed_cols[feature] = CategoryRole(object)

        return guessed_cols

    def _ok_features(self, train_data: SparkDataFrame, features: List[str]) -> List[Tuple[str, bool]]:
        """Check if column is filled well to be a feature.

        Args:
            feature: Column from dataset.

        Returns:
            ``True`` if nan ratio and freqency are not high.

        """

        row = train_data.select(
            F.count('*').alias('count'),
            *[F.mean((F.isnull(feature) | F.isnan(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
              for feature in features if isinstance(train_data.schema[feature].dataType, NumericType)],
            *[F.mean((F.isnull(feature)).astype(IntegerType())).alias(f"{feature}_nan_rate")
              for feature in features if not isinstance(train_data.schema[feature].dataType, NumericType)],
            # *[(1 - (F.approx_count_distinct(feature) / F.count(feature))).alias(f"{feature}_constant_rate")
            #   for feature in features]
        ).first()

        estimated_features = []
        for feat in features:
            if row[f"{feat}_nan_rate"] >= self.max_nan_rate:
                estimated_features.append((feat, False))
                continue

            # TODO: this part may be optimized using sampling
            crow = (
                train_data
                .groupby(feat)
                .agg(F.count('*').alias('count'))
                .select((F.max('count')).alias('count'))
                .first()
            )
            if crow['count'] / row['count'] >= self.max_constant_rate:
                estimated_features.append((feat, False))
                continue

            estimated_features.append((feat, True))

        return estimated_features

        # return [
        #     (
        #         feature,
        #         (row[f"{feature}_nan_rate"] < self.max_nan_rate)
        #         and (row[f"{feature}_constant_rate"] < self.max_constant_rate)
        #     )
        #     for feature in features
        # ]

    def read(self, data: SparkDataFrame, features_names: Any = None, add_array_attrs: bool = False) -> SparkDataset:
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like
              target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """
        kwargs = {}
        target_col = "target"
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]

                if col_name not in data.columns:
                    continue

                if array_attr == "target" and self.class_mapping is not None:
                    data = data.na.replace(self.class_mapping, subset=[target_col])

                kwargs[array_attr] = target_col

        if SparkDataset.ID_COLUMN not in data.columns:
            data = data.withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
        data = data.cache()

        data = self._create_target(data, target_col=self.target_col)
        target = data.select(SparkDataset.ID_COLUMN, self.target_col)
        kwargs["target"] = target

        def convert_column(feat: str):
            role: ColumnRole = self.roles[feat]
            if isinstance(role, DatetimeRole):
                result_column = F.to_timestamp(feat, self.roles[feat].format).alias(feat)
            elif isinstance(role, NumericRole):
                typ = dtype2Stype[role.dtype.__name__]
                result_column = (
                    F.when(F.isnull(feat), float('nan'))
                    .otherwise(F.col(feat).astype(typ))
                    .alias(feat)
                )
            else:
                result_column = F.col(feat)

            return result_column

        sdf = data
        if SparkDataset.ID_COLUMN not in data.columns:
            sdf = sdf.withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())

        sdf = sdf.select(
                SparkDataset.ID_COLUMN,
                # target_col,
                *[convert_column(feat) for feat in self.used_features]
            )

        dataset = SparkDataset(
            sdf,
            roles=self.roles,
            task=self.task,
            **kwargs
        )

        log_data("spark_reader_read", {"predict": dataset.to_pandas()})

        return dataset

    # TODO: SPARK-LAMA will be implemented later
    def advanced_roles_guess(self, dataset: SparkDataset, manual_roles: Optional[RolesDict] = None) -> RolesDict:
        """Advanced roles guess over user's definition and reader's simple guessing.

        Strategy - compute feature's NormalizedGini
        for different encoding ways and calc stats over results.
        Role is inferred by comparing performance stats with manual rules.
        Rule params are params of roles guess in init.
        Defaults are ok in general case.

        Args:
            dataset: Input PandasDataset.
            manual_roles: Dict of user defined roles.

        Returns:
            Dict.

        """
        if manual_roles is None:
            manual_roles = {}
        top_scores = []
        new_roles_dict = dataset.roles
        advanced_roles_params = deepcopy(self.advanced_roles_params)
        drop_co = advanced_roles_params.pop("drop_score_co")
        # guess roles nor numerics

        stat = get_numeric_roles_stat(
            dataset,
            manual_roles=manual_roles,
            random_state=self.random_state,
            subsample=self.samples,
        )

        if len(stat) > 0:
            # upd stat with rules

            stat = calc_encoding_rules(stat, **advanced_roles_params)
            new_roles_dict = {**new_roles_dict, **rule_based_roles_guess(stat)}
            top_scores.append(stat["max_score"])
        #
        # # # guess categories handling type
        stat = get_category_roles_stat(
            dataset,
            random_state=self.random_state,
            subsample=self.samples
        )
        if len(stat) > 0:
            # upd stat with rules
            # TODO: add sample params

            stat = calc_category_rules(stat)
            new_roles_dict = {**new_roles_dict, **rule_based_cat_handler_guess(stat)}
            top_scores.append(stat["max_score"])
        #
        # # get top scores of feature
        if len(top_scores) > 0:
            top_scores = pd.concat(top_scores, axis=0)
            # TODO: Add sample params

            null_scores = get_null_scores(
                dataset,
                list(top_scores.index),
                random_state=self.random_state,
                subsample=self.samples
            )
            top_scores = pd.concat([null_scores, top_scores], axis=1).max(axis=1)
            rejected = list(top_scores[top_scores < drop_co].index)
            logger.info("Feats was rejected during automatic roles guess: {0}".format(rejected))
            new_roles_dict = {**new_roles_dict, **{x: DropRole() for x in rejected}}

        return new_roles_dict
