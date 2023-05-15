""" Unit tests for the Exact explainer.
"""

# pylint: disable=missing-function-docstring
import pickle
import shap
from . import common


def test_interactions(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_interactions_additivity(shap.explainers.Exact, model.predict, data, data)

def test_tabular_single_output_auto_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict, data, data)

def test_tabular_multi_output_auto_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict_proba, data, data)

def test_tabular_single_output_partition_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict, shap.maskers.Partition(data), data)

def test_tabular_multi_output_partition_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict_proba, shap.maskers.Partition(data), data)

def test_tabular_single_output_independent_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict, shap.maskers.Independent(data), data)

def test_tabular_multi_output_independent_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_additivity(shap.explainers.Exact, model.predict_proba, shap.maskers.Independent(data), data)

def test_serialization(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_serialization(shap.explainers.Exact, model.predict, data, data)

def test_serialization_no_model_or_masker(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_serialization(
        shap.explainers.Exact, model.predict, data, data,
        model_saver=False, masker_saver=False,
        model_loader=lambda _: model.predict, masker_loader=lambda _: data
    )

def test_serialization_custom_model_save(basic_xgboost_scenario):
    model, data = basic_xgboost_scenario
    common.test_serialization(
        shap.explainers.Exact, model.predict, data, data,
        model_saver=pickle.dump, model_loader=pickle.load
    )
