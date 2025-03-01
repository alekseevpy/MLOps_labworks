import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_creation import import_data

X_test = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

results = y_test # results = model.predict(X_test)

accuracy = accuracy_score(y_test, y_test)
precision = precision_score(y_test, y_test)
recall = recall_score(y_test, y_test)

print("Model test accuracy is: {}".format(accuracy))
print("Model test precision is: {}".format(precision))
print("Model test recall is: {}".format(recall))
