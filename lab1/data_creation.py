import numpy as np
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


def import_data():
    all_files_present = True
    for file_name in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
      if not os.path.exists(file_name):
        all_files_present = False

    if not all_files_present:
        load_data()

    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = import_data()