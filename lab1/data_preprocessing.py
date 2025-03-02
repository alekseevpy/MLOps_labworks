import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def preprocessing_data():
    X_train = np.load("train/X.npy")
    X_test = np.load("test/X.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save("train/X_scaled.npy", X_train_scaled)
    np.save("test/X_scaled.npy", X_test_scaled)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled

preprocessing_data()
