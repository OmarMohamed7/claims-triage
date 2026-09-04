# Train the Fraud Risk Agent's classifier on the historical claims dataset.
#
# This dataset (data/raw/claims.csv) is a set of past auto insurance claims,
# one row per claim, each with a policyholder/vehicle/incident profile and a
# FraudFound_P label. 

# Run with: uv run python -m scripts.train_fraud_model
# PLAN.md, Phase 0.
#

from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ( # pyright: ignore[reportUnknownVariableType]
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import ( # pyright: ignore[reportUnknownVariableType]
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier
import joblib
import json
from sklearn.impute import SimpleImputer

from src.agents.fraud_risk import MODEL_VERSION


# load the data from the csv file
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = Path(__file__).parent.parent / "data" / "raw"
    df: pd.DataFrame = pd.read_csv(data_path / "claims.csv")
    df.drop(columns=["PolicyNumber"], inplace=True)
    target_column = "FraudFound_P"
    X: pd.DataFrame = df.drop(columns=[target_column])

    y: pd.Series = df[target_column]
    

    print(df['FraudFound_P'].value_counts())
    print("Imbalnced DataSet: FraudFound_P = 0 (No Fraud) is the majority class,  FraudFound_P = 1 (fraud) is the minority class")
    
    print(df.describe())

    return X, y


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
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
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return pipeline


def tune(pipeline: Pipeline, scale_pos_weight: float) -> RandomizedSearchCV:
    param_distributions = {
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(3, 6),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 10),
        "classifier__gamma": uniform(0, 5),
        "classifier__scale_pos_weight": uniform(
            scale_pos_weight * 0.5, scale_pos_weight
        ),
    }
    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        refit=True,
    )


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
    
    
    X_train, X_test, y_train, y_test = train_test_split( # pyright: ignore[reportUnknownVariableType]
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    scale_pos_weight = (
    len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )
    
    print(f"Len of y_train[y_train == 0]: {len(y_train[y_train == 0])}\n")
    print(f"Len of y_train[y_train == 1]: {len(y_train[y_train == 1])}\n")
    print(f"scale_pos_weight: {scale_pos_weight}\n")
    
    pipeline: Pipeline = build_pipeline(X_train, scale_pos_weight) # type: ignore

    print(y.value_counts(normalize=True))

    search = tune(pipeline, scale_pos_weight)
    search.fit(X=X_train, y=y_train) # type: ignore
    best_pipeline: Pipeline = search.best_estimator_ # type: ignore

    y_pred = best_pipeline.predict(X_test) # type: ignore
    metrics = classification_report(y_test, y_pred, output_dict=True) # type: ignore

    y_proba_all = best_pipeline.predict_proba(X_test) # type: ignore
    y_proba = y_proba_all[:, 1]  # type: ignore

    metrices: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "best_params": search.best_params_, # type: ignore
        "cv_pr_auc": search.best_score_, # type: ignore
        "classification_report": metrics,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    print(json.dumps(metrices, indent=2, default=str))

    save_model(pipeline=best_pipeline, metrics=metrices) # type: ignore


if __name__ == "__main__":
    main()
