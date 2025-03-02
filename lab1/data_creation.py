import numpy as np
import pandas as pd
import os

from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Path("train").mkdir(parents=True, exist_ok=True)
    Path("test").mkdir(parents=True, exist_ok=True)
    np.save('train/X.npy', X_train)
    np.save('test/X.npy', X_test)
    np.save('train/y.npy', y_train)
    np.save('test/y.npy', y_test)


def import_data():
    all_files_present = True
    for file_name in ['train/X.npy', 'test/X.npy', 'train/y.npy', 'test/y.npy']:
      if not os.path.exists(file_name):
        all_files_present = False

    if not all_files_present:
        load_data()

    X_train = np.load('train/X.npy')
    X_test = np.load('test/X.npy')
    y_train = np.load('train/y.npy')
    y_test = np.load('test/y.npy')
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = import_data()