import pickle  # nosec B403 - used to save (not load) the trained model artifact

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import add_engineered_features, TARGET_COLUMN

DATA_PATH = "synthetic_health_claims.csv"
MODEL_PATH = "fraud_detection_model.pkl"

df = pd.read_csv(DATA_PATH)
y = df[TARGET_COLUMN].astype(int)
X_raw = df.drop(columns=[TARGET_COLUMN])

X = add_engineered_features(X_raw)

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

feature_columns = numeric_cols + categorical_cols
X = X[feature_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ---------------------------------------------------------------------------
# Save as a .pkl file using the standard `pickle` module (not joblib), per
# the project requirement. The bundle dict keeps the trained pipeline
# together with the exact column order it expects at inference time and a
# human-readable model name -- both FastAPI and Gradio load this same file.
# ---------------------------------------------------------------------------
bundle = {
    "pipeline": pipeline,
    "model_name": "Health Claims Fraud Detector (Random Forest)",
    "model_version": "1.0",
    "feature_columns": feature_columns,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

print(f"Saved model bundle to {MODEL_PATH}")