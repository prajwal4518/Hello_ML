import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import argparse
import mlflow
import mlflow.sklearn
import json

# Define paths
INPUT_PATH = "data/processed/titanic_processed.csv"
OUTPUT_DIR = "model"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")

def train(n_estimators, max_depth):
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed data not found at {INPUT_PATH}")

    # Load data
    df = pd.read_csv(INPUT_PATH)
    
    # Split features and target
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # Save metrics
        metrics_path = "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({"accuracy": accuracy}, f)
        print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Titanic Random Forest Model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree")
    args = parser.parse_args()
    
    train(n_estimators=args.n_estimators, max_depth=args.max_depth)
