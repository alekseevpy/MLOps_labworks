import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_creation import import_data

X_test = np.load('./test/X_scaled.npy')
y_test = np.load('./test/y.npy')

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
results = model.predict(X_test)

accuracy = accuracy_score(y_test, results)
precision = precision_score(y_test, results)
recall = recall_score(y_test, results)

print("Model test accuracy is: {}".format(accuracy))
print("Model test precision is: {}".format(precision))
print("Model test recall is: {}".format(recall))
