import numpy as np
import pandas as pd

from regression_model.config import config
from regression_model.processing.validation import validate_inputs
from regression_model.processing.data_management import load_pipeline


pipeline_file_name = config.TRAINED_MODEL_DIR / "regression_model.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)
print('Persisted model loaded')


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    data = validate_inputs(input_data=data)
    prediction = _price_pipe.predict(data)
    output = np.exp(prediction)
    response = {"predictions": output}

    return response
