import sys
import warnings

import numpy as np
import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet


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
    # with mlflow.start_run():
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


if __name__ == "__main__":
    train(0.3, 0.3)
