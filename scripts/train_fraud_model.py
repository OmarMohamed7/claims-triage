# Train the Fraud Risk Agent's classifier on the historical claims dataset.
#
# This dataset (data/raw/claims.csv) is a set of past auto insurance claims,
# one row per claim, each with a policyholder/vehicle/incident profile and a
# FraudFound_P label. 

# Run with: uv run python scripts/train_fraud_model.py
# PLAN.md, Phase 0.
#

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # pyright: ignore[reportUnknownVariableType]
from sklearn.metrics import classification_report # pyright: ignore[reportUnknownVariableType]
from xgboost import XGBClassifier
import joblib
import json
from sklearn.impute import SimpleImputer


# load the data from the csv file
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = Path(__file__).parent.parent / "data" / "raw"
    df: pd.DataFrame = pd.read_csv(data_path / "claims.csv")
    df.drop(columns=["PolicyNumber"], inplace=True)
    target_column = "FraudFound_P"
    X: pd.DataFrame = df.drop(columns=[target_column])
    # get str columns to be encoded

    y: pd.Series = df[target_column]

    return X, y


def build_pipeline(X: pd.DataFrame, y) -> Pipeline:
    str_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # scale_pos_weight helps the model pay more attention to the minority class.
    '''
    Usually, your dataset might look like:

    Normal transactions (0): 9,000
    Fraud transactions  (1): 1,000
    
    So:
    len(y[y == 0]) returns: 9000
    And:
    len(y[y == 1]) returns: 1000
    Therefore:
    scale_pos_weight = 9000 / 1000
    Result:
    9.0
    
    What does scale_pos_weight = 9.0 mean?

    It tells the model:
    "Class 1 is rare, so mistakes involving class 1 should be given more importance."
    
    Without this, the model might learn:
    99% → class 0
    1%  → class 1

    and simply predict almost everything as:
    0
    It could still get high accuracy but be terrible at detecting fraud
    '''
    
    scale_pos_weight: float = len(y[y == 0]) / len(y[y == 1])

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
    joblib.dump(pipeline, ModelPath / "fraud_classifier.joblib") # type: ignore
    metrics_path = ModelPath / "fraud_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    X, y = load_data()
    pipeline: Pipeline = build_pipeline(X, y)
    
    
    X_train, X_test, y_train, y_test = train_test_split( # pyright: ignore[reportUnknownVariableType]
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    

    print(y.value_counts(normalize=True))
    pipeline.fit(X=X_train, y=y_train) # type: ignore
    y_pred = pipeline.predict(X_test) # type: ignore
    metrics = classification_report(y_test, y_pred, output_dict=True) # type: ignore
    print(json.dumps(metrics, indent=2))

    save_model(pipeline=pipeline, metrics=metrics) # type: ignore


if __name__ == "__main__":
    main()
