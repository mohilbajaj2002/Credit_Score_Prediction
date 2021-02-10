import math
from regression_model.config import config
from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_path_name=config.TESTING_DATA_FILE)
    test_data = test_data.reset_index(drop=True)
    test_data = test_data.drop('Unnamed: 0', axis = 1)
    print(test_data.head())

    single_test_json = test_data[:10].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
