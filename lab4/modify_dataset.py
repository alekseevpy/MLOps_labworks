import pandas as pd

modified = pd.read_csv("lab4/titanic_raw.csv")

modified = modified[["Pclass", "Sex", "Age"]]

modified.to_csv("lab4/titanic_raw.csv")
