from __future__ import absolute_import
from __future__ import print_function


import numpy as np



from model import TestModel

from utils import import_train_configuration, set_test_path

def H_function(next_state):
    config = import_train_configuration(config_file='training_settings.ini')
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_load'])

    # Here TestModel means LoadModel
    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )
    q = Model.predict_one(next_state)
    H = np.amax(q)

    return H

