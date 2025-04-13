import pandas as pd

modified = pd.read_csv("lab4/titanic_raw.csv")

modified = modified[["Pclass", "Sex", "Age"]]

mean_age = modified["Age"].mean()
modified["Age"] = modified["Age"].fillna(mean_age)

sex_encoded = pd.get_dummies(modified["Sex"], prefix="Sex")
modified = pd.concat([modified.drop("Sex", axis=1), sex_encoded], axis=1)

modified.to_csv("lab4/titanic_raw.csv")
