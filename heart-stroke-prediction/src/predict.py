import joblib
import numpy as np


def load_model(path):
    return joblib.load(path)


def predict_single(model, input_array):

    input_array = np.array(input_array).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_array)[0][1]
    else:
        probability = None

    return prediction, probability
