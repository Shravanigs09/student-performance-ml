"""
src/pipeline.py
Constructs a leak-free Scikit-Learn ColumnTransformer and Pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. Define feature subsets based on the Mid-Semester Warning setup
NUMERICAL_FEATURES = [
    "age",
    "Medu",          # Mother's education (0-4)
    "Fedu",          # Father's education (0-4)
    "traveltime",    # Home to school travel time
    "studytime",     # Weekly study time (1-4)
    "failures",      # Number of past class failures (0-3)
    "famrel",        # Quality of family relationships (1-5)
    "freetime",      # Free time after school (1-5)
    "goout",         # Going out with friends (1-5)
    "Dalc",          # Workday alcohol consumption (1-5)
    "Walc",          # Weekend alcohol consumption (1-5)
    "health",        # Current health status (1-5)
    "absences",      # Number of school absences (0-93)
    "G1"             # First period grade (0-20)
]

CATEGORICAL_FEATURES = [
    "school",        # 'GP' or 'MS'
    "sex",           # 'F' or 'M'
    "address",       # 'U' (urban) or 'R' (rural)
    "famsize",       # 'LE3' or 'GT3'
    "Pstatus",       # 'T' (together) or 'A' (apart)
    "Mjob",          # Mother's job
    "Fjob",          # Father's job
    "reason",        # Reason for school choice
    "guardian",      # 'mother', 'father', 'other'
    "schoolsup",     # Extra educational support
    "famsup",        # Family educational support
    "paid",          # Extra paid classes
    "activities",    # Extra-curricular activities
    "nursery",       # Attended nursery school
    "higher",        # Wants to take higher education
    "internet",      # Internet access at home
    "romantic"       # With a romantic relationship
]

def build_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer to scale continuous/ordinal variables 
    and one-hot encode categorical variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"  # Drops any columns not explicitly listed (e.g., G2, G3)
    )
    return preprocessor

def create_full_pipeline(classifier) -> Pipeline:
    """
    Combines preprocessing and an estimator into an end-to-end Pipeline.
    Prevents data leakage during cross-validation.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ]
    )