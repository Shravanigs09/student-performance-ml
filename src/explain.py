"""
src/explain.py
Extracts and plots feature importances / coefficients from the champion pipeline.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = os.path.join("models", "best_pipeline.joblib")

def explain_champion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Artifact not found at {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Retrieve feature names out of the ColumnTransformer
    num_cols = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_cols_encoded = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])

    all_features = list(num_cols) + list(cat_cols_encoded)
    coefficients = classifier.coef_[0]

    # Organize into DataFrame
    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Coefficient": coefficients,
        "Abs_Impact": abs(coefficients)
    }).sort_values(by="Abs_Impact", ascending=False)

    print("\n" + "=" * 60)
    print("      TOP 10 MOST INFLUENTIAL MODEL SIGNALS")
    print("=" * 60)
    print(importance_df[["Feature", "Coefficient"]].head(10).to_string(index=False))

    # Plot top positive and negative predictors
    top_features = pd.concat([
        importance_df.sort_values(by="Coefficient", ascending=False).head(5),
        importance_df.sort_values(by="Coefficient", ascending=True).head(5)
    ])

    plt.figure(figsize=(10, 6))
    colors = ["teal" if c > 0 else "crimson" for c in top_features["Coefficient"]]
    sns.barplot(x="Coefficient", y="Feature", data=top_features, palette=colors)
    plt.title("Top Drivers: Positive (Pass) vs Negative (At-Risk)")
    plt.xlabel("Logistic Regression Coefficient Weight")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=300)
    print("\nPlot saved to models/feature_importance.png")

if __name__ == "__main__":
    explain_champion_model()