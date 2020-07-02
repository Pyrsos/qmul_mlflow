import warnings

import numpy as np
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

remote_server_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(remote_server_uri)

mlflow.set_experiment('Boston')

def metrics(actual, pred):
    mse = np.sqrt(mean_squared_error(actual, pred))
    rmse = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return mse, rmse, r2

def load_data():
    house_dataset = datasets.load_boston()
    house_features, house_targets = house_dataset['data'], house_dataset['target']

    train_x, test_x, train_y, test_y = train_test_split(house_features, house_targets, random_state=42)

    return train_x, train_y, test_x, test_y

def train(alpha=0.5, l1_ratio=0.5):
    # train a model with given parameters
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    train_x, train_y, test_x, test_y = load_data()

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_param(key="l1_ratio", value=l1_ratio)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metrics({"mae": mae, "r2": r2})
#         mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))

        mlflow.sklearn.log_model(lr, "model")

    if __name__ == "__main__":
        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

        train(alpha, l1_ratio)
