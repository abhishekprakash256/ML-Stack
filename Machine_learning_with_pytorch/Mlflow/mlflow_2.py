"""
run tge ml flow 
"""

import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)

    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    print(f"Run ID: {run.info.run_id}")
