import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = np.load("train/X.npy")
X_test = np.load("test/X.npy")
y_train = np.load("train/y.npy")
y_test = np.load("test/y.npy")

df = pd.DataFrame(X_train)
corr = df.corrwith(pd.Series(y_train))
important_features = corr.abs().nlargest(10).index

X_train_selected = X_train[:, important_features]
X_test_selected = X_test[:, important_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

np.save("train/X_scaled.npy", X_train_scaled)
np.save("test/X_scaled.npy", X_test_scaled)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Data ppreprocessing complete.")
