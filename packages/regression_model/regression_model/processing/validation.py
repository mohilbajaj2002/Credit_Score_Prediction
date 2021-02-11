from regression_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()
    cols = validated_data.columns

    # check for variables with string dtypes not seen during training
    for c in cols:
        if (validated_data[c].dtype == object):
            validated_data = validated_data.drop([ç], axis = 1)

    return validated_data
