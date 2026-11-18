"""
app.py
10/10 Production Streamlit App:
- Multi-subject model routing (Mathematics vs Portuguese)
- Human-readable ordinal scales
- Per-student local feature attribution chart
- Dynamic decision threshold tuning
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Academic Risk Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATHS = {
    "Mathematics (student-mat)": os.path.join("models", "math_pipeline.joblib"),
    "Portuguese Language (student-por)": os.path.join("models", "por_pipeline.joblib"),
}

@st.cache_resource
def load_pipeline(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at {model_path}. Run `python src/train.py` first!")
        st.stop()
    return joblib.load(model_path)


def compute_local_contributions(pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    """Computes per-student feature contributions (transformed_value * coef)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    x_vec = X_trans[0]

    num_cols = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.transformers_[1][1]
    cat_cols = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    feature_names = num_cols + cat_cols

    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
        contributions = x_vec * weights
    else:
        # Fallback if a tree model won CV
        contributions = x_vec * classifier.feature_importances_

    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions,
        "Abs": np.abs(contributions)
    }).sort_values(by="Abs", ascending=False)

    return df_contrib.head(8)


EDU_LABELS = {0: "0 - None", 1: "1 - Primary (4th grade)", 2: "2 - Middle (5th-9th)", 3: "3 - Secondary (High School)", 4: "4 - Higher Education"}
TRAVEL_LABELS = {1: "1 - < 15 min", 2: "2 - 15-30 min", 3: "3 - 30-60 min", 4: "4 - > 1 hour"}
QUALITY_LABELS = {1: "1 - Very Bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}
INTENSITY_LABELS = {1: "1 - Very Low", 2: "2 - Low", 3: "3 - Moderate", 4: "4 - High", 5: "5 - Very High"}

st.title("🎓 Student Academic Risk & Early-Warning System")
st.markdown("Predicts academic failure risk ($G3 < 10$) after Period 1 exams with per-student risk attribution.")

# Subject Selector
selected_course = st.selectbox("Select Course Cohort Model", list(MODEL_PATHS.keys()))
pipeline = load_pipeline(MODEL_PATHS[selected_course])

st.divider()
st.subheader("Core Academic & Behavioral Indicators")

col1, col2 = st.columns(2)
with col1:
    g1 = st.slider("Period 1 Grade (G1: 0 - 20)", 0, 20, 11)
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
    absences = st.number_input("Total School Absences", min_value=0, max_value=93, value=4)
    studytime = st.selectbox(
        "Weekly Study Time", [1, 2, 3, 4], index=1,
        format_func=lambda x: {1: "1 - < 2 hrs", 2: "2 - 2 to 5 hrs", 3: "3 - 5 to 10 hrs", 4: "4 - > 10 hrs"}[x]
    )

with col2:
    age = st.slider("Student Age", 15, 22, 17)
    higher = st.selectbox("Wants Higher Education?", ["yes", "no"], index=0)
    schoolsup = st.selectbox("Extra School Support?", ["no", "yes"], index=0)
    romantic = st.selectbox("In a Romantic Relationship?", ["no", "yes"], index=0)

with st.expander("⚙️ Optional Demographic & Lifestyle Factors (Pre-filled Defaults)"):
    opt1, opt2 = st.columns(2)
    with opt1:
        guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"], index=0)
        famsup = st.selectbox("Family Educational Support", ["yes", "no"], index=0)
        paid = st.selectbox("Extra Paid Classes", ["no", "yes"], index=0)
        activities = st.selectbox("Extracurricular Activities", ["yes", "no"], index=0)
        internet = st.selectbox("Home Internet Access", ["yes", "no"], index=0)
        goout = st.selectbox("Going Out with Friends", [1, 2, 3, 4, 5], index=2, format_func=lambda x: INTENSITY_LABELS[x])
    with opt2:
        medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=2, format_func=lambda x: EDU_LABELS[x])
        fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], index=2, format_func=lambda x: EDU_LABELS[x])
        traveltime = st.selectbox("Travel Time to School", [1, 2, 3, 4], index=0, format_func=lambda x: TRAVEL_LABELS[x])
        famrel = st.selectbox("Family Relationship Quality", [1, 2, 3, 4, 5], index=3, format_func=lambda x: QUALITY_LABELS[x])
        freetime = st.selectbox("Free Time After School", [1, 2, 3, 4, 5], index=2, format_func=lambda x: INTENSITY_LABELS[x])
        walc = st.selectbox("Weekend Alcohol Consumption", [1, 2, 3, 4, 5], index=0, format_func=lambda x: INTENSITY_LABELS[x])

st.divider()
threshold = st.slider("Intervention Threshold (Lower = higher sensitivity for at-risk students)", 0.20, 0.80, 0.50, step=0.05)

if st.button("Evaluate Student Risk Profile", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "G1": g1, "failures": failures, "absences": absences, "studytime": studytime,
        "age": age, "higher": higher, "schoolsup": schoolsup, "romantic": romantic,
        "guardian": guardian, "famsup": famsup, "paid": paid, "activities": activities,
        "internet": internet, "goout": goout, "Medu": medu, "Fedu": fedu,
        "traveltime": traveltime, "famrel": famrel, "freetime": freetime, "Walc": walc,
        "school": "GP", "sex": "F", "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Mjob": "other", "Fjob": "other", "reason": "course",
        "nursery": "yes", "Dalc": 1, "health": 4
    }])

    probs = pipeline.predict_proba(input_df)[0]
    p_fail, p_pass = probs[0], probs[1]
    is_passing = p_pass >= threshold

    st.subheader("Assessment Result")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if is_passing:
            st.success("### ✅ Good Standing (Pass)")
        else:
            st.error("### ⚠️ At-Risk (Intervention Needed)")
        st.metric("Pass Probability", f"{p_pass * 100:.1f}%")

    with res_col2:
        st.metric("At-Risk Probability", f"{p_fail * 100:.1f}%")
        st.progress(float(p_pass))
        if not is_passing:
            st.warning(f"Pass probability ({p_pass*100:.1f}%) is below the {threshold*100:.0f}% threshold.")

    # Local Explainability Chart
    st.subheader("🔍 Why Did the Model Make This Prediction?")
    st.caption("Positive values push the student toward **Passing**; negative values push toward **At-Risk**.")
    contrib_df = compute_local_contributions(pipeline, input_df).set_index("Feature")
    st.bar_chart(contrib_df["Contribution"])