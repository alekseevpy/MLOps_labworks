import pandas as pd

modified = pd.read_csv('./titanic_raw.csv')

modified = modified[['Pclass', 'Sex', 'Age']]

modified.to_csv('titanic_raw.csv')
