import pathlib
import pandas as pd
import regression_model
from scipy.stats import normaltest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'

ALL_VARS = []
CATEGORICAL_VARS = []
NUMERICAL_VARS = []
NUMERICAL_LOG_VARS = []
alpha = 0.05

df = pd.read_csv(str(DATASET_DIR) + r"\dataset.csv")
cols = df.columns
cols = cols[:-1]
ALL_VARS = cols.to_list()

# Creating files for categorical and numerical features
f = open(str(DATASET_DIR) + r"\cat_vars.txt", "w")
g = open(str(DATASET_DIR) + r"\num_vars.txt", "w")
for c in cols:
    if (df[c].nunique() <= 15):
        CATEGORICAL_VARS.append(c)
        f.write(c)
        f.write(' ')
    else:
        NUMERICAL_VARS.append(c)
        g.write(c)
        g.write(' ')

f.close()
g.close()


# Creating file for log normal features
f = open(str(DATASET_DIR) + r"\num_log_vars.txt", "w")
for num in NUMERICAL_VARS:
    if (any(df[num] <= 0)):
        print(f'Feature {num}, contains negative values. Moving on... ')
    else:
        k2, p = normaltest(df[num].values)
        if (p < alpha):
            NUMERICAL_LOG_VARS.append(num)
            f.write(num)
            f.write(' ')

f.close()

# Creating file for all features
f = open(str(DATASET_DIR) + r"\feature_list.txt", "w")
for c in cols:
    f.write(c)
    f.write(' ')

f.close()




""""
for c in cols:
    if (df[c].nunique() <= 15):
        CATEGORICAL_VARS.append(c)
    else:
        NUMERICAL_VARS.append(c)


for num in NUMERICAL_VARS:
    if(any(df[num]<=0)):
        continue
    else:
      k2, p = normaltest(df[num].values)
      if (p < alpha):
          NUMERICAL_LOG_VARS.append(num)


f = open(str(DATASET_DIR) + r"\feature_list.txt", "w")
f.write(str(ALL_VARS))
f.close()

f = open(str(DATASET_DIR) + r"\cat_vars.txt", "w")
f.write(str(CATEGORICAL_VARS))
f.close()

f = open(str(DATASET_DIR) + r"\num_vars.txt", "w")
f.write(str(NUMERICAL_VARS))
f.close()

f = open(str(DATASET_DIR) + r"\num_log_vars.txt", "w")
f.write(str(NUMERICAL_LOG_VARS))
f.close()
"""
