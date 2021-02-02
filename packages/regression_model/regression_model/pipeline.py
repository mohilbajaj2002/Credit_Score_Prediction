from sklearn.pipeline import Pipeline
import pandas as pd
import preprocessors as pp

CATEGORICAL_VARS = []
df = pd.read_csv("datasets/dataset.csv")
cols = df.columns

for c in cols:
    if (df[c].nunique() <= 15):
        CATEGORICAL_VARS.append(c)

train = df[:int(0.75*len(df))]
test = df[int(0.75*len(df)):]

f = open("datasets/feature_list.txt", "w")
f.write(str(cols))
f.close()

train.to_csv("datasets/train.csv")
test.to_csv("datasets/test.csv")

PIPELINE_NAME = 'lasso_regression'

price_pipe = Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=CATEGORICAL_VARS)),
    ])
