import numpy as np
from scipy.stats import normaltest
from regression_model.config import config
from sklearn.base import BaseEstimator, TransformerMixin


def FileReader(file_path):
    f = open(file_path, "r")
    data = f.read()
    f.close()

    return data



class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables_path=None):

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        print('Class LogTransformer Instantiated')


    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check if the list is empty
        if (len(self.variables) == 0):
            print('Feature list empty. Moving on...')

        else:
            # check that the values are non-negative for log transform
            for num in self.variables:
                if(any(X[num]<=0)):
                    print (f'Feature {num}, contains negative values. Moving on...')
                else:
                    k2, p = normaltest(X[num].values)
                    if (p < config.alpha):
                        X[num] = X[num].apply(lambda x: np.log(x))

        print('4 - Log Transformation done')
        return X
