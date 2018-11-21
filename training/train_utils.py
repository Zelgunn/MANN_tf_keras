from keras import Model
import keras.utils
import os
from time import time


def get_log_dir(base_dir):
    return os.path.join(base_dir, "log_{0}".format(int(time())))


def save_model_json(model: Model, log_dir):
    filename = "{0}_config.json".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        file.write(model.to_json())


def save_model_summary(model: Model, log_dir):
    filename = "{0}_summary.txt".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))


def save_model_info(model: Model, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    keras.utils.plot_model(model, os.path.join(log_dir, "{0}.png".format(model.name)))
    save_model_json(model, log_dir)
    save_model_summary(model, log_dir)
