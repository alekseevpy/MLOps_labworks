import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def preprocessing_data():
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save("X_train_scaled.npy", X_train_scaled)
    np.save("X_test_scaled.npy", X_test_scaled)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled

preprocessing_data()
