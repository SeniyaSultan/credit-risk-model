import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -----------------------
# Data Loading
# -----------------------
def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]
    return X, y


def get_feature_columns(df: pd.DataFrame):
    """Separate numeric and categorical columns"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove CustomerId if present
    numeric_cols = [c for c in numeric_cols if c != "CustomerId"]
    categorical_cols = [c for c in categorical_cols if c != "CustomerId"]
    
    return numeric_cols, categorical_cols


# -----------------------
# Model Pipeline
# -----------------------
def build_model_pipeline(model, numeric_cols, categorical_cols):
    """Build preprocessing + model pipeline with missing value handling"""
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline


# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }


# -----------------------
# Training and Logging
# -----------------------
def train_and_log(model_pipeline, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model_pipeline.fit(X_train, y_train)
        
        metrics = evaluate_model(model_pipeline, X_test, y_test)
        
        # Log model params and metrics
        mlflow.log_params(model_pipeline.named_steps["classifier"].get_params())
        mlflow.log_metrics(metrics)
        
        # Use `name` instead of deprecated `artifact_path`
        mlflow.sklearn.log_model(model_pipeline, name=model_name)
        
        return metrics


# -----------------------
# Main
# -----------------------
def main():
    # Set tracking URI (optional, ensures all runs go to a known location)
    mlflow.set_tracking_uri("file:///C:/Users/jkk/OneDrive/Desktop/credit-risk-model/mlruns")
    
    # Ensure experiment exists
    experiment_name = "Credit Risk Model"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Load data
    X, y = load_data("data/processed/processed_data.csv")
    numeric_cols, categorical_cols = get_feature_columns(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        pipeline = build_model_pipeline(model, numeric_cols, categorical_cols)
        results[name] = train_and_log(pipeline, name, X_train, X_test, y_train, y_test)
    
    print("✅ Training completed. Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()
