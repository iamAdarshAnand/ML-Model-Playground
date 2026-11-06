import pytest
from ml_core import DATASETS, MODELS, train_evaluate, load_dataset, make_model

@pytest.mark.parametrize("dataset_name", list(DATASETS))
@pytest.mark.parametrize("model_name", MODELS)
def test_train_evaluate(dataset_name, model_name):
    result = train_evaluate(dataset_name, model_name)
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.macro_f1 <= 1
    assert result.confusion.sum() == len(result.truth)
    assert len(result.predictions) == len(result.truth)

def test_invalid_dataset():
    with pytest.raises(ValueError):
        load_dataset("unknown")

def test_invalid_model():
    with pytest.raises(ValueError):
        make_model("unknown")

def test_invalid_split():
    with pytest.raises(ValueError):
        train_evaluate("Iris", "Random Forest", 0.9)
