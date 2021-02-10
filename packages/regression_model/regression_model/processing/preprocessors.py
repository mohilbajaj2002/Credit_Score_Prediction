import numpy as np
import pandas as pd
from scipy.stats import normaltest
from sklearn.base import BaseEstimator, TransformerMixin
from regression_model.config import config


def FileReader(file_path):
    f = open(file_path, "r")
    data = f.read()
    f.close()

    return data


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables_path=None) -> None:

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        print('Class CategoricalImputer Instantiated')


    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        flag = 0
        X = X.copy()
        for feature in self.variables:
            try:
                mode = X[feature].mode(dropna=True)[0]

            except:
                mode = 0

            X[feature] = X[feature].fillna(mode)

        for feature in self.variables:
            if not(X[feature].count() == len(X)):
                flag = 1

        if (flag == 1):
            transform(X)
        else:
            print('1 - Categorical Imputation done')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables_path=None):

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        print('Class NumericalImputer Instantiated')

    def fit(self, X, y=None):
        # persist mean in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):

        flag = 0
        X = X.copy()
        for feature in self.variables:
            try:
                X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            except:
                X[feature].fillna(0)

        for feature in self.variables:
            if not(X[feature].count() == len(X)):
                flag = 1

        if (flag == 1):
            transform(X)
        else:
            print('2 - Numerical Imputation done')

        return X



class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables_path=None):

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")

        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        print('Class RareLabelCategoricalEncoder Instantiated')

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )
            X[feature] = X[feature].replace('Rare', X[feature].mode()[0])


        print('3 - Rare Label Categorical Encoding done')
        return X



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


class StandardizeNumericalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_path=None):

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")
        self.variables = variables

        print('Class StandardizeNumericalFeatures Instantiated')


    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature] = pd.to_numeric(X[feature])
            mean = X[feature].mean()
            std = X[feature].std()
            X[feature] = X[feature].apply(lambda x: (x-mean)/std)

        print('5 - Numerical Feature Standardization done')
        return X



class BestFeaturesSelection(BaseEstimator, TransformerMixin):
    def __init__(self, variables_path=None):

        variables = FileReader(variables_path)
        variables = str(variables).strip().split(" ")
        self.variables = variables

        print('Class BestFeaturesSelection Instantiated')


    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        X = X[self.variables]

        print('6 - Best Feature Selection done')
        return X
