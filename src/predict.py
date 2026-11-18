"""
src/predict.py
Runs inference on new student profiles using the serialized pipeline artifact.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "best_pipeline.joblib")

def load_inference_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run src/train.py first.")
    return joblib.load(MODEL_PATH)

def predict_student_risk(student_data: pd.DataFrame, decision_threshold: float = 0.50):
    """
    Predicts student outcome with custom probability thresholding.
    Lowering threshold increases sensitivity for detecting at-risk students.
    """
    pipeline = load_inference_pipeline()
    
    # Predict probabilities: index 0 = At-Risk, index 1 = Pass
    probabilities = pipeline.predict_proba(student_data)
    pass_probs = probabilities[:, 1]
    
    results = []
    for idx, p_pass in enumerate(pass_probs):
        # If pass probability is below threshold, flag as at-risk
        status = "Pass" if p_pass >= decision_threshold else "At-Risk / Needs Intervention"
        results.append({
            "Student_ID": idx + 1,
            "Predicted_Status": status,
            "Pass_Probability": f"{p_pass * 100:.1f}%",
            "At_Risk_Probability": f"{(1.0 - p_pass) * 100:.1f}%"
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test with two contrasting synthetic test profiles
    sample_students = pd.DataFrame([
        {
            # Profile 1: High risk (low G1, past backlogs, high absences)
            "school": "GP", "sex": "M", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 1, "Fedu": 1, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
            "traveltime": 2, "studytime": 1, "failures": 2, "schoolsup": "no", "famsup": "no",
            "paid": "no", "activities": "no", "nursery": "no", "higher": "no", "internet": "no",
            "romantic": "no", "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 2, "Walc": 3,
            "health": 2, "absences": 12, "G1": 6
        },
        {
            # Profile 2: Safe standing (solid G1, good study habits, zero backlogs)
            "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "LE3", "Pstatus": "T",
            "Medu": 4, "Fedu": 3, "Mjob": "health", "Fjob": "services", "reason": "reputation", "guardian": "mother",
            "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes",
            "paid": "yes", "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
            "romantic": "no", "famrel": 4, "freetime": 2, "goout": 2, "Dalc": 1, "Walc": 1,
            "health": 5, "absences": 2, "G1": 15
        }
    ])
    
    print("\n--- Running Inference via Saved Model Pipeline ---")
    predictions = predict_student_risk(sample_students, decision_threshold=0.50)
    print(predictions.to_string(index=False))