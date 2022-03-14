"""Classes for report generation and add-ons."""

import logging
import math
import os
import warnings

from copy import copy
from copy import deepcopy
from operator import itemgetter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jinja2 import Environment
from jinja2 import FileSystemLoader
from json2html import json2html
from pyspark import RDD
from pyspark.mllib.linalg import DenseMatrix
from pyspark.sql import SparkSession

from pyspark.sql.dataframe import DataFrame
from pyspark.mllib.evaluation import BinaryClassificationMetrics, RegressionMetrics, MulticlassMetrics
from pyspark.mllib.stat import KernelDensity
import pyspark.sql.functions as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



from lightautoml.addons.uplift import metrics as uplift_metrics
from lightautoml.addons.uplift.metalearners import TLearner
from lightautoml.addons.uplift.metalearners import XLearner
from lightautoml.addons.uplift.utils import _get_treatment_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.automl.presets.tabular_presets import ReadableIntoSparkDf
from lightautoml.spark.report.handy_spark_utils import call2
from lightautoml.spark.transformers.scala_wrappers.laml_string_indexer import LAMLStringIndexer, LAMLStringIndexerModel

logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)


def extract_params(input_struct):
    params = dict()
    iterator = input_struct if isinstance(input_struct, dict) else input_struct.__dict__
    for key in iterator:
        if key.startswith(("_", "autonlp_params")):
            continue
        value = iterator[key]
        if type(value) in [bool, int, float, str]:
            params[key] = value
        elif value is None:
            params[key] = None
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            params[key] = extract_params(value)
        else:
            params[key] = str(type(value))
    return params


def get_data_for_roc_and_pr_curve(input_data, scores_col_name="raw", true_labels_col_name="labels"):
    data = round_score_col(input_data=input_data,
                           min_co=0.001,
                           max_co=0.999,
                           step=0.001,
                           scores_col_name=scores_col_name,
                           true_labels_col_name=true_labels_col_name)

    metrics = BinaryClassificationMetrics(
        scoreAndLabels=data.select(
            F.col(f"{scores_col_name}_rounded").astype("double"),
            F.col(true_labels_col_name).astype("double")
        ).rdd
    )

    thresholds = call2(metrics, "thresholds").collect()
    roc = call2(metrics, "roc").collect()
    pr = call2(metrics, "pr").collect()

    df = pd.DataFrame(
        list(
            zip(thresholds, map(itemgetter(0), roc), map(itemgetter(1), roc), map(itemgetter(1), pr))
        ) + [(0., 1., 1., 0.)]
    )
    df.columns = ["thresholds", "fpr", "recall", "precision"]

    auc_score = metrics.areaUnderROC
    ap_score = metrics.areaUnderPR

    return [df, auc_score, ap_score]


def plot_curves(input_data, scores_col_name, positive_rate, true_labels_col_name, roc_path, pr_path):
    rounded_data, auc_score, ap_score = get_data_for_roc_and_pr_curve(input_data, scores_col_name, true_labels_col_name)

    plot_roc_curve_image(rounded_data, auc_score, roc_path)
    plot_pr_curve_image(rounded_data, ap_score, positive_rate, pr_path)
    return auc_score, ap_score, rounded_data


def round_score_col(input_data, min_co=0.01, max_co=0.99, step=0.01,
                    scores_col_name="raw", true_labels_col_name="labels"):
    scores = F.col(scores_col_name)
    _id_col = F.col(SparkDataset.ID_COLUMN)
    true_labels = F.col(true_labels_col_name)

    return input_data.select(
        _id_col,
        true_labels,
        (F.ceil(scores / step) * step).alias("_scores_prepared_value")
    ).select(
        _id_col,
        true_labels,
        F.when(
            F.col("_scores_prepared_value") < min_co,
            min_co
        ).otherwise(
            F.when(
                F.col("_scores_prepared_value") > max_co,
                max_co
            ).otherwise(
                F.col("_scores_prepared_value")
            )
        ).alias(f"{scores_col_name}_rounded")
    )


def plot_roc_curve_image(data, auc_score, path):

    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    fpr = data["fpr"]
    tpr = data["recall"]

    lw = 2
    plt.plot(fpr, tpr, color="blue", lw=lw, label="Trained model")
    plt.plot([0, 1], [0, 1], color="red", lw=lw, linestyle="--", label="Random model")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color="gray", linestyle="-", linewidth=1)
    plt.title("Approx ROC curve (GINI = {:.3f})".format(2 * auc_score - 1))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_pr_curve_image(data, ap_score, positive_rate, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    precision = data["precision"]
    recall = data["recall"]

    lw = 2
    plt.plot(recall, precision, color="blue", lw=lw, label="Trained model")
    plt.plot(
        [0, 1],
        [positive_rate, positive_rate],
        color="red",
        lw=lw,
        linestyle="--",
        label="Random model",
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.45, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color="gray", linestyle="-", linewidth=1)
    plt.title("Approx PR curve (AP = {:.3f})".format(ap_score))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_preds_distribution_by_bins(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    box_plot_data = []
    labels = []
    for name, group in data.groupby("bin"):
        labels.append(name)
        box_plot_data.append(group["y_pred"].values)

    box = axs.boxplot(box_plot_data, patch_artist=True, labels=labels)
    for patch in box["boxes"]:
        patch.set_facecolor("green")
    axs.set_yscale("log")
    axs.set_xlabel("Bin number")
    axs.set_ylabel("Prediction")
    axs.set_title("Distribution of object predictions by bin")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_distribution_of_logits(input_data, path, scores_col_name, true_labels_col_name):

    prep_data = round_score_col(input_data=input_data,
                           min_co=0.001,
                           max_co=0.999,
                           step=0.001,
                           scores_col_name=scores_col_name,
                           true_labels_col_name=true_labels_col_name)

    logits_col_name = f"{scores_col_name}_rounded"
    logits_col = F.col(logits_col_name)

    data = prep_data.select(logits_col).groupby(logits_col).count().toPandas()

    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    data["proba_logit"] = np.log(data[logits_col_name].values / (1 - data[logits_col_name].values))

    data_0 = data[data[true_labels_col_name] == 0]
    sns.kdeplot(
        data_0["proba_logit"],
        shade=True,
        color="r",
        label="Class 0 logits",
        ax=axs,
        weights=data_0["count"]
    )

    data_1 = data[data[true_labels_col_name] == 1]
    sns.kdeplot(
        data_1["proba_logit"],
        shade=True,
        color="g",
        label="Class 1 logits",
        ax=axs,
        weights=data_1["count"]
    )
    axs.set_xlabel("Logits")
    axs.set_ylabel("Density")
    axs.set_title("Logits distribution of object predictions (by classes)")
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_pie_f1_metric(data: RDD, path):
    metrics = MulticlassMetrics(predictionAndLabels=data)
    tn, fp, fn, tp = metrics.confusionMatrix().values
    F1 = metrics.fMeasure(1.0)
    prec = metrics.precision(1.0)
    rec = metrics.recall(1.0)

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

    recipe = [
        "{} True Positives".format(tp),
        "{} False Positives".format(fp),
        "{} False Negatives".format(fn),
        "{} True Negatives".format(tn),
    ]

    wedges, texts = ax.pie([tp, fp, fn, tn], wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(
        arrowprops=dict(arrowstyle="-", color="k"),
        bbox=bbox_props,
        zorder=0,
        va="center",
    )

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(
            recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y), horizontalalignment=horizontalalignment, **kw
        )

    ax.set_title(
        "Trained model: Precision = {:.2f}%, Recall = {:.2f}%, F1-Score = {:.2f}%".format(
            prec * 100, rec * 100, F1 * 100
        )
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return prec, rec, F1


def f1_score_w_co(input_data, min_co=0.01, max_co=0.99, step=0.01,
                  true_labels_col_name="labels", scores_col_name="raw"):

    true_labels = F.col(true_labels_col_name)
    scores = F.col(scores_col_name)

    data = round_score_col(input_data=input_data,
                           min_co=min_co,
                           max_co=max_co,
                           step=step,
                           true_labels_col_name=true_labels_col_name,
                           scores_col_name=scores_col_name)

    _grp = data.groupby(scores).agg(
        F.sum(true_labels).alias("sum"),
        F.count(true_labels).alias("count")
    ).toPandas()
    pos = _grp["sum"].sum()
    neg = _grp["count"].sum() - pos

    positive_rate = pos / _grp["count"].sum()

    grp = _grp.groupby("y_pred").agg(sum=("sum", "sum"), count=("count", "sum"))
    grp.sort_index(inplace=True)

    grp["fp"] = grp["sum"].cumsum()
    grp["tp"] = pos - grp["fp"]
    grp["tn"] = (grp["count"] - grp["sum"]).cumsum()
    grp["fn"] = neg - grp["tn"]

    grp["pr"] = grp["tp"] / (grp["tp"] + grp["fp"])
    grp["rec"] = grp["tp"] / (grp["tp"] + grp["fn"])

    grp["f1_score"] = 2 * (grp["pr"] * grp["rec"]) / (grp["pr"] + grp["rec"])

    best_score = grp["f1_score"].max()
    best_co = grp.index.values[grp["f1_score"] == best_score].mean()

    return best_score, best_co, positive_rate


def get_bins_table(data: DataFrame, n_bins=20):
    df: pd.DataFrame = data.groupby(
        F.col("y_pred"), F.col("y_true")
    ).count().toPandas().sort_values(
        by=["y_pred", "y_true"],
        ascending=[False, False]
    )

    total_count = df["count"].sum()
    bin_size = math.ceil(total_count / n_bins)




    bins_table = data.groupby("bin").agg({"y_true": [len, np.mean], "y_pred": [np.min, np.mean, np.max]}).reset_index()
    bins_table.columns = [
        "Bin number",
        "Amount of objects",
        "Mean target",
        "Min probability",
        "Average probability",
        "Max probability",
    ]
    return bins_table.to_html(index=False)


# Regression plots:


def plot_target_distribution_1(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(2, 1, figsize=(16, 20))

    sns.kdeplot(data["y_true"], shade=True, color="g", ax=axs[0])
    axs[0].set_xlabel("Target value")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Target distribution (y_true)")

    sns.kdeplot(data["y_pred"], shade=True, color="r", ax=axs[1])
    axs[1].set_xlabel("Target value")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Target distribution (y_pred)")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_target_distribution_2(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    sns.kdeplot(data["y_true"], shade=True, color="g", label="y_true", ax=axs)
    sns.kdeplot(data["y_pred"], shade=True, color="r", label="y_pred", ax=axs)
    axs.set_xlabel("Target value")
    axs.set_ylabel("Density")
    axs.set_title("Target distribution")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_target_distribution(data, path):
    data_pred = pd.DataFrame({"Target value": data["y_pred"]})
    data_pred["source"] = "y_pred"
    data_true = pd.DataFrame({"Target value": data["y_true"]})
    data_true["source"] = "y_true"
    data = pd.concat([data_pred, data_true], ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.displot(
        data,
        x="Target value",
        row="source",
        height=9,
        aspect=1.5,
        kde=True,
        color="m",
        facet_kws=dict(margin_titles=True),
    )
    g.fig.suptitle("Target distribution")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_error_hist(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.kdeplot(data["y_pred"] - data["y_true"], shade=True, color="m", ax=ax)
    ax.set_xlabel("Error = y_pred - y_true")
    ax.set_ylabel("Density")
    ax.set_title("Error histogram")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_reg_scatter(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.jointplot(
        x="y_pred",
        y="y_true",
        data=data,
        kind="reg",
        truncate=False,
        color="m",
        height=14,
    )
    g.fig.suptitle("Scatter plot")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches="tight")
    plt.close()


# Multiclass plots:


def plot_confusion_matrix(data: DenseMatrix, path):
    arr = data.toArray()
    cmat: "np.array" = arr / arr.sum(axis=1, keepdims=True)
    # cmat: "np.array" = confusion_matrix(data["y_true"], data["y_pred"], normalize="true")
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(cmat, annot=True, linewidths=0.5, cmap="Purples", ax=ax)
    ax.set_xlabel("y_pred")
    ax.set_ylabel("y_true")
    ax.set_title("Confusion matrix")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


# Feature importance


def plot_feature_importance(feat_imp, path, features_max=100):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, features_max / 2.5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp[:features_max], ax=axs, color="m")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


class ReportDeco:
    """
    Decorator to wrap :class:`~lightautoml.automl.base.AutoML` class to generate html report on ``fit_predict`` and ``predict``.

    Example:

        >>> report_automl = ReportDeco(output_path="output_path", report_file_name="report_file_name")(automl).
        >>> report_automl.fit_predict(train_data)
        >>> report_automl.predict(test_data)

    Report will be generated at output_path/report_file_name automatically.

    Warning:
         Do not use it just to inference (if you don"t need report), because:

            - It needs target variable to calc performance metrics.
            - It takes additional time to generate report.
            - Dump of decorated automl takes more memory to store.

    To get unwrapped fitted instance to pickle
    and inferecne access ``report_automl.model`` attribute.

    """

    @property
    def model(self):
        """Get unwrapped model.

        Returns:
            model.

        """
        return self._model

    @property
    def mapping(self):
        return self._model.reader.class_mapping

    @property
    def task(self):
        return self._model.reader.task._name

    def __init__(self, *args, **kwargs):
        """

        Note:
            Valid kwargs are:

                - output_path: Folder with report files.
                - report_file_name: Name of main report file.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        """
        if not kwargs:
            kwargs = {}

        # default params
        self.fi_params = {"method": "fast", "n_sample": 100_000}
        self.interpretation_params = {
            "top_n_features": 5,
            "top_n_categories": 10,
            "ton_n_classes": 10,
            "n_bins": 30,
            "datetime_level": "year",
            "n_sample": 100_000,
        }

        fi_input_params = kwargs.get("fi_params", {})
        self.fi_params.update(fi_input_params)
        interpretation_input_params = kwargs.get("interpretation_params", {})
        self.interpretation_params.update(interpretation_input_params)
        self.interpretation = kwargs.get("interpretation", False)

        self.n_bins = kwargs.get("n_bins", 20)
        self.template_path = kwargs.get("template_path", os.path.join(base_dir, "lama_report_templates/"))
        self.output_path = kwargs.get("output_path", "lama_report/")
        self.report_file_name = kwargs.get("report_file_name", "lama_interactive_report.html")
        self.pdf_file_name = kwargs.get("pdf_file_name", None)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self._base_template_path = "lama_base_template.html"
        self._model_section_path = "model_section.html"
        self._train_set_section_path = "train_set_section.html"
        self._results_section_path = "results_section.html"
        self._fi_section_path = "feature_importance_section.html"
        self._interpretation_section_path = "interpretation_section.html"
        self._interpretation_subsection_path = "interpretation_subsection.html"

        self._inference_section_path = {
            "binary": "binary_inference_section.html",
            "reg": "reg_inference_section.html",
            "multiclass": "multiclass_inference_section.html",
        }

        self.title = "Spark-LAMA report"
        if self.interpretation:
            self.sections_order = [
                "intro",
                "model",
                "train_set",
                "fi",
                "interpretation",
                "results",
            ]
            self._interpretation_top = []
        else:
            self.sections_order = ["intro", "model", "train_set", "fi", "results"]
        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []

        self.generate_report()

    def __call__(self, model):
        self._model = model

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self

    def _binary_classification_details(self, data, positive_rate,
                                       true_labels_col_name, scores_col_name, predicted_labels_col_name):
        # self._inference_content["sample_bins_table"] = get_bins_table(data)
        # plot_preds_distribution_by_bins(
        #     data,
        #     path=os.path.join(self.output_path, self._inference_content["preds_distribution_by_bins"]),
        # )

        # Done
        prec, rec, F1 = plot_pie_f1_metric(
            data.select(
                F.col(predicted_labels_col_name).astype("double"),
                F.col(true_labels_col_name).astype("double")
            ).rdd,
            path=os.path.join(self.output_path, self._inference_content["pie_f1_metric"]),
        )

        # Done
        auc_score, ap_score, rounded_data = plot_curves(
            input_data=data,
            scores_col_name=scores_col_name,
            positive_rate=positive_rate,
            true_labels_col_name=true_labels_col_name,
            roc_path=os.path.join(self.output_path, self._inference_content["roc_curve"]),
            pr_path=os.path.join(self.output_path, self._inference_content["pr_curve"])
        )

        # Done
        plot_distribution_of_logits(
            input_data=data,
            path=os.path.join(self.output_path, self._inference_content["distribution_of_logits"]),
            scores_col_name=scores_col_name,
            true_labels_col_name=true_labels_col_name
        )

        return auc_score, prec, rec, F1

    def _regression_details(self, data, true_values_col_name, predictions_col_name):
        # graphics
        # TODO
        # plot_target_distribution(
        #     data,
        #     path=os.path.join(self.output_path, self._inference_content["target_distribution"]),
        # )
        # plot_error_hist(
        #     data,
        #     path=os.path.join(self.output_path, self._inference_content["error_hist"]),
        # )
        # plot_reg_scatter(
        #     data,
        #     path=os.path.join(self.output_path, self._inference_content["scatter_plot"]),
        # )
        # metrics
        metrics = RegressionMetrics(
            predictionAndObservations=data.select(
                F.col(predictions_col_name).astype("double"),
                F.col(true_values_col_name).astype("double")
            ).rdd
        )

        mean_ae = metrics.meanAbsoluteError
        # mean_ae = mean_absolute_error(data["y_true"], data["y_pred"])

        median_ae = data.select(
            F.percentile_approx(
                F.abs(
                    F.col(predictions_col_name).astype("double") - F.col(true_values_col_name).astype("double")
                ),
                0.5
            )
        ).first()[0]
        # median_ae = median_absolute_error(data["y_true"], data["y_pred"])

        mse = metrics.meanSquaredError
        # mse = mean_squared_error(data["y_true"], data["y_pred"])

        r2 = metrics.r2
        # r2 = r2_score(data["y_true"], data["y_pred"])

        evs = metrics.explainedVariance
        # evs = explained_variance_score(data["y_true"], data["y_pred"])

        return mean_ae, median_ae, mse, r2, evs

    def _multiclass_details(self, data, predicted_labels_col_name, true_labels_col_name):

        true_labels_col = F.col(true_labels_col_name)

        metrics = MulticlassMetrics(
            predictionAndLabels=data.select(
                F.col(predicted_labels_col_name).astype("double"),
                true_labels_col.astype("double")
            ).rdd
        )

        # tn, fp, fn, tp = metrics.confusionMatrix().values
        labels_counts = data.select(true_labels_col).groupby(true_labels_col).count().toPandas()
        total_labels_count = labels_counts["count"].sum()

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

        p_micro = metrics.accuracy  # TODO: ???
        p_macro_sum = 0.
        for label in labels_counts[true_labels_col_name]:
            p_macro_sum += metrics.precision(float(label))
        p_macro = p_macro_sum / total_labels_count
        p_weighted = metrics.weightedPrecision

        r_micro = metrics.accuracy  # TODO: ???
        r_macro_sum = 0.
        for label in labels_counts[true_labels_col_name]:
            r_macro_sum += metrics.recall(float(label))
        r_macro = r_macro_sum / total_labels_count
        r_weighted = metrics.weightedRecall

        f_micro = metrics.accuracy  # TODO: ???
        f_macro_sum = 0.
        for label in labels_counts[true_labels_col_name]:
            f_macro_sum += metrics.fMeasure(float(label))
        f_macro = f_macro_sum / total_labels_count
        f_weighted = metrics.weightedFMeasure()

        # y_true = data["y_true"]
        # y_pred = data["y_pred"]
        # precision
        # p_micro = precision_score(y_true, y_pred, average="micro")
        # p_macro = precision_score(y_true, y_pred, average="macro")
        # p_weighted = precision_score(y_true, y_pred, average="weighted")
        # recall
        # r_micro = recall_score(y_true, y_pred, average="micro")
        # r_macro = recall_score(y_true, y_pred, average="macro")
        # r_weighted = recall_score(y_true, y_pred, average="weighted")
        # f1-score
        # f_micro = f1_score(y_true, y_pred, average="micro")
        # f_macro = f1_score(y_true, y_pred, average="macro")
        # f_weighted = f1_score(y_true, y_pred, average="weighted")

        # classification report for features
        if self.mapping:
            classes = sorted(self.mapping, key=self.mapping.get)
        else:
            classes = np.arange(self._N_classes)

        p = [metrics.precision(label) for label in labels_counts[true_labels_col_name]]
        r = [metrics.recall(label) for label in labels_counts[true_labels_col_name]]
        f = [metrics.fMeasure(label) for label in labels_counts[true_labels_col_name]]
        s = list(labels_counts["count"])

        # p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        cls_report = pd.DataFrame(
            {
                "Class name": classes,
                "Precision": p,
                "Recall": r,
                "F1-score": f,
                "Support": s,
            }
        )
        self._inference_content["classification_report"] = cls_report.to_html(
            index=False, float_format="{:.4f}".format, justify="left"
        )

        plot_confusion_matrix(
            metrics.confusionMatrix(),
            path=os.path.join(self.output_path, self._inference_content["confusion_matrix"]),
        )

        return [
            p_micro,
            p_macro,
            p_weighted,
            r_micro,
            r_macro,
            r_weighted,
            f_micro,
            f_macro,
            f_weighted,
        ]

    def _collect_data(self,
                      preds: SparkDataset,
                      sample,
                      true_values_col_name,
                      raw_predictions_col_name,
                      predicted_labels_col_name) -> DataFrame:
        # predict_column = preds.features[0]
        data = preds.data.join(
            sample,  # sample.select(F.col(SparkDataset.ID_COLUMN), F.col(self._target)),
            on=SparkDataset.ID_COLUMN
        ).select(
            F.col(SparkDataset.ID_COLUMN),
            F.col(self._target).alias(true_values_col_name),
            F.col(raw_predictions_col_name).alias(raw_predictions_col_name),
            F.col(predicted_labels_col_name).alias(predicted_labels_col_name)
        ).where(
            (~F.isnan(F.col(predicted_labels_col_name))) & (F.col(predicted_labels_col_name).isNotNull())
        )
        # TODO SPARK-LAMA: Create an UDF to map values for multiclass task

        # data = pd.DataFrame({"y_true": sample[self._target].values})
        # if self.task in "multiclass":
        #     if self.mapping is not None:
        #         data["y_true"] = np.array([self.mapping[y] for y in data["y_true"].values])
        #     data["y_pred"] = preds._data.argmax(axis=1)
        #     data = data[~np.isnan(preds._data).any(axis=1)]
        # else:
        #     data["y_pred"] = preds._data[:, 0]
        #     data.sort_values("y_pred", ascending=False, inplace=True)
        #     data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
        #     data = data[~data["y_pred"].isnull()]
        return data

    def fit_predict(self, *args, **kwargs):
        """Wrapped ``automl.fit_predict`` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        # TODO: parameters parsing in general case

        preds: SparkDataset = self._model.fit_predict(*args, **kwargs)

        csv_df = pd.read_csv("/mnt/hgfs/Projects/Sber/LAMA/Sber-LAMA-Stuff/dumps/labels_preds.csv")
        csv_df = csv_df[["y_true", "y_pred", "label"]]
        csv_df.columns = ["y_true", "raw", "label"]
        preds = SparkSession.builder.getOrCreate().createDataFrame(csv_df)

        true_values_col_name = "y_true"
        scores_col_name = "raw"
        predictions_col_name = "label"

        train_data: DataFrame = kwargs["train_data"] if "train_data" in kwargs else args[0]
        input_roles = kwargs["roles"] if "roles" in kwargs else args[1]
        self._target = input_roles["target"]
        valid_data: Optional[DataFrame] = kwargs.get("valid_data", None)

        if valid_data is None:
            data = self._collect_data(
                preds, train_data, true_values_col_name, scores_col_name, predictions_col_name
            )
        else:
            data = self._collect_data(
                preds, valid_data, true_values_col_name, scores_col_name, predictions_col_name
            )

        self._inference_content = {}
        if self.task == "binary":
            # filling for html
            self._inference_content = {}
            self._inference_content["roc_curve"] = "valid_roc_curve.png"
            self._inference_content["pr_curve"] = "valid_pr_curve.png"
            self._inference_content["pie_f1_metric"] = "valid_pie_f1_metric.png"
            self._inference_content["preds_distribution_by_bins"] = "valid_preds_distribution_by_bins.png"
            self._inference_content["distribution_of_logits"] = "valid_distribution_of_logits.png"
            # graphics and metrics
            _, self._F1_thresh, positive_rate = f1_score_w_co(data)
            auc_score, prec, rec, F1 = self._binary_classification_details(
                data,
                positive_rate,
                true_labels_col_name=true_values_col_name,
                scores_col_name=scores_col_name,
                predicted_labels_col_name=predictions_col_name
            )
            # update model section
            evaluation_parameters = ["AUC-score", "Precision", "Recall", "F1-score"]
            self._model_summary = pd.DataFrame(
                {
                    "Evaluation parameter": evaluation_parameters,
                    "Validation sample": [auc_score, prec, rec, F1],
                }
            )
        elif self.task == "reg":
            # filling for html
            self._inference_content["target_distribution"] = "valid_target_distribution.png"
            self._inference_content["error_hist"] = "valid_error_hist.png"
            self._inference_content["scatter_plot"] = "valid_scatter_plot.png"
            # graphics and metrics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data, true_values_col_name, predictions_col_name)
            # model section
            evaluation_parameters = [
                "Mean absolute error",
                "Median absolute error",
                "Mean squared error",
                "R^2 (coefficient of determination)",
                "Explained variance",
            ]
            self._model_summary = pd.DataFrame(
                {
                    "Evaluation parameter": evaluation_parameters,
                    "Validation sample": [mean_ae, median_ae, mse, r2, evs],
                }
            )
        elif self.task == "multiclass":
            self._N_classes = len(train_data[self._target].drop_duplicates())
            self._inference_content["confusion_matrix"] = "valid_confusion_matrix.png"

            index_names = np.array([["Precision", "Recall", "F1-score"], ["micro", "macro", "weighted"]])
            index = pd.MultiIndex.from_product(index_names, names=["Evaluation metric", "Average"])

            summary = self._multiclass_details(data, predictions_col_name, true_values_col_name)
            self._model_summary = pd.DataFrame({"Validation sample": summary}, index=index)

        self._inference_content["title"] = "Results on validation sample"

        self._generate_model_section()

        # generate train data section
        self._train_data_overview = self._data_genenal_info(train_data)
        self._describe_roles(train_data)
        self._describe_dropped_features(train_data)
        self._generate_train_set_section()
        # generate fit_predict section
        self._generate_inference_section()

        # generate feature importance and interpretation sections
        # self._generate_fi_section(valid_data)

        # if self.interpretation:
        #     self._generate_interpretation_section(valid_data)

        self.generate_report()
        return preds

    def predict(self, *args, **kwargs):
        """Wrapped automl.predict method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: arguments.
            **kwargs: additional parameters.

        Returns:
            predictions.

        """
        self._n_test_sample += 1
        # get predictions
        test_preds = self._model.predict(*args, **kwargs)

        test_data = kwargs["test"] if "test" in kwargs else args[0]
        data = self._collect_data(test_preds, test_data)

        if self.task == "binary":
            # filling for html
            self._inference_content = {}
            self._inference_content["roc_curve"] = "test_roc_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pr_curve"] = "test_pr_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pie_f1_metric"] = "test_pie_f1_metric_{}.png".format(self._n_test_sample)
            self._inference_content["bins_preds"] = "test_bins_preds_{}.png".format(self._n_test_sample)
            self._inference_content["preds_distribution_by_bins"] = "test_preds_distribution_by_bins_{}.png".format(
                self._n_test_sample
            )
            self._inference_content["distribution_of_logits"] = "test_distribution_of_logits_{}.png".format(
                self._n_test_sample
            )
            # graphics and metrics
            # positive_rate = data.where()
            auc_score, prec, rec, F1 = self._binary_classification_details(data)

            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = [
                    auc_score,
                    prec,
                    rec,
                    F1,
                ]
            else:
                self._model_summary["Test sample"] = [auc_score, prec, rec, F1]

        elif self.task == "reg":
            # filling for html
            self._inference_content = {}
            self._inference_content["target_distribution"] = "test_target_distribution_{}.png".format(
                self._n_test_sample
            )
            self._inference_content["error_hist"] = "test_error_hist_{}.png".format(self._n_test_sample)
            self._inference_content["scatter_plot"] = "test_scatter_plot_{}.png".format(self._n_test_sample)
            # graphics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data)
            # update model section
            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = [
                    mean_ae,
                    median_ae,
                    mse,
                    r2,
                    evs,
                ]
            else:
                self._model_summary["Test sample"] = [mean_ae, median_ae, mse, r2, evs]

        elif self.task == "multiclass":
            self._inference_content["confusion_matrix"] = "test_confusion_matrix_{}.png".format(self._n_test_sample)
            test_summary = self._multiclass_details(data)
            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = test_summary
            else:
                self._model_summary["Test sample"] = test_summary

        # layout depends on number of test samples
        if self._n_test_sample >= 2:
            self._inference_content["title"] = "Results on test sample {}".format(self._n_test_sample)

        else:
            self._inference_content["title"] = "Results on test sample"

        # update model section
        self._generate_model_section()
        # generate predict section
        self._generate_inference_section()

        self.generate_report()
        return test_preds

    # TODO SPARK-LAMA: Required method _model.get_feature_scores is not implemented for Spark.
    def _generate_fi_section(self, valid_data: Optional[DataFrame]):
        total_count = valid_data.count()
        if (
            self.fi_params["method"] == "accurate"
            and valid_data is not None
            and total_count > self.fi_params["n_sample"]
        ):
            valid_data = valid_data.sample(n=self.fi_params["n_sample"])
            print(
                "valid_data was sampled for feature importance calculation: n_sample = {}".format(
                    self.fi_params["n_sample"]
                )
            )

        if self.fi_params["method"] == "accurate" and valid_data is None:
            # raise ValueError("You must set valid_data with accurate feature importance method")
            self.fi_params["method"] = "fast"
            warnings.warn(
                "You must set valid_data with 'accurate' feature importance method. Changed to 'fast' automatically."
            )

        self.feat_imp = self._model.get_feature_scores(
            calc_method=self.fi_params["method"], data=valid_data, silent=False
        )
        if self.feat_imp is None:
            fi_path = None
        else:
            fi_path = "feature_importance.png"
            plot_feature_importance(self.feat_imp, path=os.path.join(self.output_path, fi_path))
        # add to _sections
        fi_content = {
            "fi_method": self.fi_params["method"],
            "feature_importance": fi_path,
        }
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        fi_section = env.get_template(self._fi_section_path).render(fi_content)
        self._sections["fi"] = fi_section

    # TODO SPARK-LAMA: Required method _model.get_individual_pdp is not implemented for Spark.
    def _generate_interpretation_content(self, test_data):
        self._interpretation_content = {}
        if test_data is None:
            self._interpretation_content["interpretation_top"] = None
            return
        if self.feat_imp is None:
            interpretation_feat_list = list(self._model.reader._roles.keys())[
                : self.interpretation_params["top_n_features"]
            ]
        else:
            interpretation_feat_list = self.feat_imp["Feature"].values[: self.interpretation_params["top_n_features"]]
        for feature_name in interpretation_feat_list:
            interpretaton_subsection = {}
            interpretaton_subsection["feature_name"] = feature_name
            interpretaton_subsection["feature_interpretation_plot"] = feature_name + "_interpretation.png"
            self._plot_pdp(
                test_data,
                feature_name,
                path=os.path.join(
                    self.output_path,
                    interpretaton_subsection["feature_interpretation_plot"],
                ),
            )
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            interpretation_subsection = env.get_template(self._interpretation_subsection_path).render(
                interpretaton_subsection
            )
            self._interpretation_top.append(interpretation_subsection)
            print(f"Interpretation info for {feature_name} appended")
        self._interpretation_content["interpretation_top"] = self._interpretation_top

    def _generate_interpretation_section(self, test_data):
        if test_data is not None and test_data.shape[0] > self.interpretation_params["n_sample"]:
            test_data = test_data.sample(n=self.interpretation_params["n_sample"])
        self._generate_interpretation_content(test_data)
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        interpretation_section = env.get_template(self._interpretation_section_path).render(
            self._interpretation_content
        )
        self._sections["interpretation"] = interpretation_section

    def _plot_pdp(self, test_data, feature_name, path):
        feature_role = self._model.reader._roles[feature_name].name
        # I. Count interpretation
        print("Calculating interpretation for {}:".format(feature_name))
        grid, ys, counts = self._model.get_individual_pdp(
            test_data=test_data,
            feature_name=feature_name,
            n_bins=self.interpretation_params["n_bins"],
            top_n_categories=self.interpretation_params["top_n_categories"],
            datetime_level=self.interpretation_params["datetime_level"],
        )
        # II. Plot pdp
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
        axs[0].set_title("PDP plot: " + feature_name)
        n_classes = ys[0].shape[1]
        if n_classes == 1:
            data = pd.concat(
                [pd.DataFrame({"x": grid[i], "y": ys[i].ravel()}) for i, _ in enumerate(grid)]
            ).reset_index(drop=True)
            if feature_role in ["Numeric", "Datetime"]:
                g0 = sns.lineplot(data=data, x="x", y="y", ax=axs[0], color="m")
            else:
                g0 = sns.boxplot(data=data, x="x", y="y", ax=axs[0], showfliers=False, color="m")
        else:
            if self.mapping:
                classes = sorted(self.mapping, key=self.mapping.get)[: self.interpretation_params["top_n_classes"]]
            else:
                classes = np.arange(min(n_classes, self.interpretation_params["top_n_classes"]))
            data = pd.concat(
                [
                    pd.DataFrame({"x": grid[i], "y": ys[i][:, k], "class": name})
                    for i, _ in enumerate(grid)
                    for k, name in enumerate(classes)
                ]
            ).reset_index(drop=True)
            if self._model.reader._roles[feature_name].name in ["Numeric", "Datetime"]:
                g0 = sns.lineplot(data=data, x="x", y="y", hue="class", ax=axs[0])
            else:
                g0 = sns.boxplot(data=data, x="x", y="y", hue="class", ax=axs[0], showfliers=False)
        g0.set(ylabel="y_pred")
        # III. Plot distribution
        counts = np.array(counts) / sum(counts)
        if feature_role == "Numeric":
            g0.set(xlabel="feature value")
            g1 = sns.histplot(test_data[feature_name], kde=True, color="gray", ax=axs[1])
        elif feature_role == "Category":
            g0.set(xlabel=None)
            axs[0].set_xticklabels(grid, rotation=90)
            g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
        else:
            g0.set(xlabel=self.interpretation_params["datetime_level"])
            g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
        g1.set(xlabel=None)
        g1.set(ylabel="Frequency")
        g1.set(xticklabels=[])
        # IV. Save picture
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close()

    def _data_genenal_info(self, data):
        general_info = pd.DataFrame(columns=["Parameter", "Value"])
        general_info.loc[0] = ("Number of records", data.count())
        general_info.loc[1] = ("Total number of features", len(data.columns))
        general_info.loc[2] = ("Used features", len(self._model.reader._used_features))
        general_info.loc[3] = (
            "Dropped features",
            len(self._model.reader._dropped_features),
        )
        # general_info.loc[4] = ("Number of positive cases", np.sum(data[self._target] == 1))
        # general_info.loc[5] = ("Number of negative cases", np.sum(data[self._target] == 0))
        return general_info.to_html(index=False, justify="left")

    def _describe_roles(self, train_data):

        # detect feature roles
        roles = self._model.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]

        total_count = train_data.count()

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            current_column = F.col(feature_name)  # current_column
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(
                train_data.select(current_column).where(
                    F.isnan(current_column) | F.isnull(current_column)
                ).count()
                / total_count
            )
            values: DataFrame = train_data.where(~F.isnan(current_column) & ~F.isnull(current_column))

            stat_data: list = values.select(
                F.min(current_column),
                F.percentile_approx(current_column, 0.25),
                F.avg(current_column),
                F.percentile_approx(current_column, 0.5),
                F.percentile_approx(current_column, 0.75),
                F.max(current_column)
            ).first()

            item["min"] = stat_data[0]
            item["quantile_25"] = stat_data[1]
            item["average"] = stat_data[2]
            item["median"] = stat_data[3]
            item["quantile_75"] = stat_data[4]
            item["max"] = stat_data[5]
            numerical_features_df.append(item)
        if numerical_features_df == []:
            self._numerical_features_table = None
        else:
            self._numerical_features_table = pd.DataFrame(numerical_features_df).to_html(
                index=False, float_format="{:.2f}".format, justify="left"
            )

        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            current_column = F.col(feature_name)
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(
                train_data.select(current_column).where(
                    F.isnan(current_column) | F.isnull(current_column)
                ).count()
                / total_count
            )
            value_counts = train_data.select(current_column).groupby(current_column).count().select(
                current_column,
                F.col("count") / total_count  # normalization
            ).toPandas()
            values = value_counts[feature_name]
            counts = value_counts["count"]
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurance of most frequent"] = "{:.1f}%".format(100 * counts[0])
            item["Least frequent value"] = values[-1]
            item["Occurance of least frequent"] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        if categorical_features_df == []:
            self._categorical_features_table = None
        else:
            self._categorical_features_table = pd.DataFrame(categorical_features_df).to_html(
                index=False, justify="left"
            )
        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            current_column = F.col(feature_name)
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(
                train_data.select(current_column).where(
                    F.isnan(current_column) | F.isnull(current_column)
                ).count()
                / total_count
            )
            values = train_data.where(~F.isnan(current_column) & ~F.isnull(current_column))

            stat_data = values.select(
                F.min(current_column),
                F.max(current_column)
            ).first()

            item["min"] = stat_data[0]
            item["max"] = stat_data[1]
            item["base_date"] = self._model.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        if datetime_features_df == []:
            self._datetime_features_table = None
        else:
            self._datetime_features_table = pd.DataFrame(datetime_features_df).to_html(index=False, justify="left")

    def _describe_dropped_features(self, train_data):
        total_count = train_data.count()
        self._max_nan_rate = self._model.reader.max_nan_rate
        self._max_constant_rate = self._model.reader.max_constant_rate
        self._features_dropped_list = self._model.reader._dropped_features
        # dropped features table
        dropped_list = [col for col in self._features_dropped_list if col != self._target]
        if dropped_list == []:
            self._dropped_features_table = None
        else:
            dropped_nan_ratio = train_data.select(
                *[
                    (F.sum(
                        F.when(
                            F.isnan(F.col(col_name)), 1
                        ).otherwise(
                            F.when(
                                F.isnull(F.col(col_name)), 1
                            ).otherwise(0)
                        )
                    ) / total_count).alias(col_name)
                    for col_name in dropped_list
                ]
            ).toPandas()
            # dropped_nan_ratio = train_data[dropped_list].isna().sum() / train_data.shape[0]
            dropped_most_occured = pd.Series(np.nan, index=dropped_list)

            indexer = LAMLStringIndexer(
                inputCols=dropped_list,
                outputCols=[f"{col}_out" for col in dropped_list],
                minFreqs=[0 for _ in dropped_list],
                handleInvalid="keep",
                defaultValue=0.,
                freqLabel=True
            )

            indexer_model: LAMLStringIndexerModel = indexer.fit(train_data)

            encodings = indexer_model.labelsArray  # list of string tuples ('key', 'count'), example: ('key', '11')
            dropped_most_occured = [
                int(sorted(enc, key=lambda x: int(x[1]), reverse=True)[0][1]) / total_count
                for enc in encodings
            ]

            # for col in dropped_list:
            #     col_most_occured = train_data[col].value_counts(normalize=True).values
            #     if len(col_most_occured) > 0:
            #         dropped_most_occured[col] = col_most_occured[0]
            dropped_features_table = pd.DataFrame(
                {"nan_rate": dropped_nan_ratio, "constant_rate": dropped_most_occured}
            )
            self._dropped_features_table = (
                dropped_features_table.reset_index()
                .rename(columns={"index": "Название переменной"})
                .to_html(index=False, justify="left")
            )

    def _generate_model_section(self):
        model_summary = None
        if self._model_summary is not None:
            model_summary = self._model_summary.to_html(
                index=self.task == "multiclass",
                justify="left",
                float_format="{:.4f}".format,
            )

        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        model_section = env.get_template(self._model_section_path).render(
            model_name=self._model_name,
            model_parameters=self._model_parameters,
            model_summary=model_summary,
        )
        self._sections["model"] = model_section

    def _generate_train_set_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        train_set_section = env.get_template(self._train_set_section_path).render(
            train_data_overview=self._train_data_overview,
            numerical_features_table=self._numerical_features_table,
            categorical_features_table=self._categorical_features_table,
            datetime_features_table=self._datetime_features_table,
            target=self._target,
            max_nan_rate=self._max_nan_rate,
            max_constant_rate=self._max_constant_rate,
            dropped_features_table=self._dropped_features_table,
        )
        self._sections["train_set"] = train_set_section

    def _generate_inference_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        inference_section = env.get_template(self._inference_section_path[self.task]).render(self._inference_content)
        self._model_results.append(inference_section)

    def _generate_results_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            results_section = env.get_template(self._results_section_path).render(model_results=self._model_results)
            self._sections["results"] = results_section

    def generate_report(self):
        # collection sections
        self._generate_results_section()
        sections_list = []
        for sec_name in self.sections_order:
            if sec_name in self._sections:
                sections_list.append(self._sections[sec_name])
        # put sections inside
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        report = env.get_template(self._base_template_path).render(
            title=self.title, sections=sections_list, pdf=self.pdf_file_name
        )

        with open(os.path.join(self.output_path, self.report_file_name), "w", encoding="utf-8") as f:
            f.write(report)

        if self.pdf_file_name:
            try:
                from weasyprint import HTML

                HTML(string=report, base_url=self.output_path).write_pdf(
                    os.path.join(self.output_path, self.pdf_file_name)
                )
            except ModuleNotFoundError:
                print("Can't generate PDF report: check manual for installing pdf extras.")
