#!/usr/bin/env python

import argparse
import joblib
import os
import sys
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import sagemaker_mlflow  # noqa: F401
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    num_cpus = os.environ.get("SM_NUM_CPUS", 4)
    parser.add_argument("--n_jobs", type=int, default=num_cpus)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=120)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def start(args):
    """
    Train a Random Forest Classifier
    """

    # Load example dataset (Pima Indians Diabetes, binary classification).
    # OpenML dataset id 37; features are 8 numeric clinical measurements and
    # the target is tested_positive / tested_negative for diabetes.
    X, y = fetch_openml(data_id=37, as_frame=False, return_X_y=True)

    # Encode the string target labels into 0/1 for the classifier
    y = LabelEncoder().fit_transform(y)

    # Alternatively read data from os.environ.get("SM_INPUT_DATA_DIR")

    # Split into train and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    hyperparameters = {
        "max_depth": args.max_depth,
        "verbose": 1,
        "n_jobs": args.n_jobs,
        "n_estimators": args.n_estimators,
    }

    model = RandomForestClassifier()
    model.set_params(**hyperparameters)
    model.fit(X_train, y_train)

    # Metrics we care about
    accuracy = model.score(X_validation, y_validation)
    print("accuracy: {}".format(accuracy))
    f1 = f1_score(y_validation, model.predict(X_validation))
    print("F1: {}".format(f1))

    # Save the model
    joblib.dump(model, os.path.join(os.environ["SM_MODEL_DIR"], "model.joblib"))


if __name__ == "__main__":
    args, _ = parse_args()

    # Experiment tracking is optional. The ARN is passed on the command line as a
    # string, so a disabled run arrives as "None" (Python None interpolated into
    # the command) or "" — treat those as "skip tracking". Gate on the ARN alone;
    # the experiment name is only meaningful once a tracking server is set.
    mlflow_arn = (args.mlflow_arn or "").strip()
    if mlflow_arn and mlflow_arn.lower() != "none":
        mlflow.set_tracking_uri(mlflow_arn)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.autolog()
        # Tag SageMaker AI Training job name so we can use it to pull
        # experiment data later if we register the model
        mlflow.set_tag("sm_training_job_name", os.environ.get("TRAINING_JOB_NAME"))
    else:
        logger.info("No MLflow ARN provided; skipping experiment tracking.")

    start(args)
