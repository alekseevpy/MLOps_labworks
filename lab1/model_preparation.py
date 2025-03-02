import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.load("train/X_scaled.npy")
y_train = np.load("train/y.npy")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
