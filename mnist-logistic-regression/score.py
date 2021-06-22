import json
import numpy as np
import pandas as pd
import os
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

mnist_model = None


def init():
    global mnist_model
    try:
        mnist_model_file = Model.get_model_path('mnist')
        #print('Series Model root:', mnist_model_root)
        #mnist_model_file = os.path.join(mnist_model_root, 'outputs/model.sav')
        print('Series Model file:', mnist_model_file)
        mnist_model = pickle.load(open(mnist_model_file, 'rb'))
    except Exception as e:
        print(e)

input_sample = np.random.rand(2, 784)
output_sample = np.array([[0], [1]])


@input_schema('data', NumpyParameterType(input_sample, enforce_shape=False))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    print(type(data))
    log_data({"data shape": str(data.shape)})

    data = np.array(data)
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(data)
    data = scaling.transform(data)

    log_data({"data shape after scaling": str(data.shape)})

    # make prediction
    numbers_predicted = mnist_model.predict(data)
    log_data({"predictions": str(numbers_predicted)})

    return numbers_predicted.tolist()


def log_data(logging_data: dict):
    print(json.dumps(logging_data))
