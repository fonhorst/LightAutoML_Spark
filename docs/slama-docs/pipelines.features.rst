.. role:: hidden
    :class: hidden-section


lightautoml.spark.pipelines.features
==============================

Pipelines for features generation.

Base Classes
-----------------

.. currentmodule:: lightautoml.spark.pipelines.features.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkFeaturesPipeline
    SparkTabularDataFeatures
    SparkEmptyFeaturePipeline
    SparkNoOpTransformer
    SelectTransformer
    FittedPipe
    build_graph



Feature Pipelines for Boosting Models
-----------------------------------------

.. currentmodule:: lightautoml.spark.pipelines.features.lgb_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLGBSimpleFeatures
    SparkLGBAdvancedPipeline


Feature Pipelines for Linear Models
-----------------------------------

.. currentmodule:: lightautoml.spark.pipelines.features.linear_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLinearFeatures
