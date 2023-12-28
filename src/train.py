import argparse
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

from unet.random_selection import *
from pre_processing.convert import *
from pre_processing.data_augmentation import *
from training.train import *
from testing.predict import *
from testing.output import *
from testing.output import *


def train(train_images, train_masks, args):
    model = train_model(train_images, train_masks)
    return model


def save_model(model, args):
    # model_dir is /opt/ml/model/
    model_output = os.path.join(args.model_dir, "model.joblib")
    print(f"Saving model to {model_output}")
    joblib.dump(model, model_output)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--test", type=str, default="/opt/ml/input/data/test")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--inspect", type=bool, default=False)
    args, _ = parser.parse_known_args()
    print(f"Received arguments {args}")
    return args


def main(args):
    train_images, train_masks = args.train_images, args.train_masks
    model = train(train_images, train_masks, args)
    save_model(model, args)


if __name__ == "__main__":
    args = parse_arg()
    if args.inspect:
        os.environ["PYTHONINSPECT"] = "1"
    main(args)