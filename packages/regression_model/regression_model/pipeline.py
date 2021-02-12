from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import features
from regression_model.processing import preprocessors as pp
from regression_model.config import config

import logging


_logger = logging.getLogger(__name__)



price_pipe = Pipeline(
    [
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables_path=config.CATEGORICAL_VARS_FILE),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables_path=config.NUMERICAL_VARS_FILE),
        ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tol=0.01, variables_path=config.CATEGORICAL_VARS_FILE),
        ),
        (
            "log_transformer",
            features.LogTransformer(variables_path=config.NUMERICAL_LOG_VARS_FILE)
        ),
        (
            "scaler",
            MinMaxScaler(),
        ),
        (
            "Linear_model",
            Lasso(alpha=0.005, random_state=0)
        ),
    ]
)
