import argparse
import os
import tarfile
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
try:
    from sklearn.externals import joblib
except:
    import joblib

from unet.random_selection import *
from pre_processing.convert import *
from pre_processing.data_augmentation import *


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer")
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--data-dir", type=str, default="opt/ml/processing")
    parser.add_argument("--data-input", type=str, default="input/census-income.csv")
    args, _ = parser.parse_known_args()
    print(f"Received arguments {args}")
    return args


def main(args):
    input_data_path = os.path.join(args.data_dir, args.data_input)

    if args.mode == "infer":
        train_images, train_masks, val_images, val_masks = read_random_training_and_validation_data()
        return val_images, val_masks
    
    elif args.mode == "train":
        apply_dataset_masking()

        train_images, train_masks, val_images, val_masks = read_random_training_and_validation_data()
        train_images, train_masks = apply_image_augmentation(train_images, train_masks)
        val_images, val_masks = apply_image_augmentation(val_images, val_masks)

        return train_images, train_masks, val_images, val_masks


if __name__ == "__main__":
    args = parse_arg()
    main(args)