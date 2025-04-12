from catboost.datasets import titanic

train_df, _ = titanic()

train_df.to_csv("lab4/titanic_raw.csv", index=False)
