"""Shared fixtures for explainer tests"""

import sys
import pytest

import shap


@pytest.fixture(scope="session")
def basic_xgboost_scenario():
    """ Create a basic XGBoost model on a data set.
    """
    dataset=shap.datasets.adult
    max_samples=100
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = dataset()
    if max_samples is not None:
        X = X.iloc[:max_samples]
        y = y[:max_samples]
    X = X.values

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    return model, X


@pytest.mark.skipif(sys.platform == 'win32', reason="Integer division bug in HuggingFace on Windows")
@pytest.fixture(scope="session")
def basic_translation_scenario():
    """ Create a basic transformers translation model and tokenizer.
    """
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    AutoModelForSeq2SeqLM = pytest.importorskip("transformers").AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    # define the input sentences we want to translate
    data = [
        "In this picture, there are four persons: my father, my mother, my brother and my sister.",
        "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models"
    ]

    return model, tokenizer, data
