import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def preprocessing_data():
    X_train = np.load("lab1/train/X_train.npy")
    X_test = np.load("lab1/test/X_test.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save("lab1/train/X_train_scaled.npy", X_train_scaled)
    np.save("lab1/test/X_test_scaled.npy", X_test_scaled)

    with open("lab1/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled

preprocessing_data()