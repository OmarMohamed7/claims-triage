# Train the Fraud Risk Agent's classifier on the historical claims dataset.
#
# This dataset (data/raw/claims.csv) is a set of past auto insurance claims,
# one row per claim, each with a policyholder/vehicle/incident profile and a
# FraudFound_P label. In this project it does double duty: it trains the
# model here, and at inference time (src/agents/fraud_risk.py) the same file
# is used as a mock policyholder lookup table keyed by PolicyNumber, standing
# in for a real policy/claims-history system this demo doesn't have.
#
# Run with: uv run python scripts/train_fraud_model.py
# See PLAN.md, Phase 0.
#

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import json
from sklearn.impute import SimpleImputer


# TODO: load_data() -> (X, y)
# - Read data/raw/claims.csv.
# - y = the FraudFound_P column.
# - X = everything else, minus PolicyNumber (it's a row identifier used for
#   the lookup table at inference, not a predictive feature — including it
#   would let the model memorize rows instead of learning from the actual
#   profile/incident fields).


# load the data from the csv file
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "raw"
    df = pd.read_csv(data_path / "claims.csv")
    df.drop(columns=["PolicyNumber"], inplace=True)
    target_column = "FraudFound_P"
    X = df.drop(columns=[target_column])
    # get str columns to be encoded

    y = df[target_column]

    return X, y


#
# TODO: build_pipeline(X, y) -> sklearn Pipeline
# - ColumnTransformer: one-hot encode the categorical (object dtype)
#   columns, passthrough the numeric columns.
# - Dataset is heavily imbalanced (~6% fraud) — weight the positive class
#   (e.g. XGBClassifier's scale_pos_weight = n_n
# eg / n_pos) so the model
#   doesn't just learn to predict "not fraud" every time.
# - Wrap preprocessing + classifier in a single Pipeline so joblib can
#   serialize the whole thing as one artifact.
#


def build_pipeline(X, y):
    str_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    transformers=[
                        ("num", num_pipeline, num_cols),
                        (
                            "encoder",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                            str_cols,
                        ),
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )
    return pipeline


# TODO: main()
# - Train/test split (stratified on y, since the classes are imbalanced).
# - Fit the pipeline on the training split.
# - Evaluate on the test split with precision / recall / f1 / roc_auc /
#   pr_auc on the fraud class — NOT accuracy (a model predicting the
#   majority class "not fraud" every time would score >94% accuracy while
#   catching 0 fraud cases).
# - Save the fitted pipeline to models/fraud_classifier.joblib
#   (joblib.dump) and the metrics to models/fraud_classifier_metrics.json.


def save_model(pipeline: Pipeline, metrics: dict):
    ModelPath = Path(__file__).parent.parent / "models"
    if not ModelPath.exists():
        ModelPath.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ModelPath / "fraud_classifier.joblib")
    metrics_path = ModelPath / "fraud_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    X, y = load_data()
    pipeline = build_pipeline(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(y.value_counts(normalize=True))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    print(json.dumps(metrics, indent=2))

    save_model(pipeline=pipeline, metrics=metrics)


if __name__ == "__main__":
    main()
