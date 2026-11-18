"""
tests/test_pipeline.py
Automated unit tests verifying model artifact loading, schema validation,
and probability output bounds.
"""

import os
import joblib
import pandas as pd
import pytest

MODEL_PATHS = [
    os.path.join("models", "math_pipeline.joblib"),
    os.path.join("models", "por_pipeline.joblib"),
]

SAMPLE_RECORD = pd.DataFrame([{
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
    "Walc": 1, "health": 4, "absences": 4, "G1": 12
}])


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_pipeline_inference(model_path):
    assert os.path.exists(model_path), f"Missing trained artifact: {model_path}"
    pipeline = joblib.load(model_path)

    probs = pipeline.predict_proba(SAMPLE_RECORD)
    assert probs.shape == (1, 2), "Output shape must be (1, 2) for binary classification"
    assert abs(probs.sum() - 1.0) < 1e-6, "Class probabilities must sum to 1.0"
    assert 0.0 <= probs[0][1] <= 1.0, "Pass probability must lie in [0, 1]"