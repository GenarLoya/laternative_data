# This is only expample of how to save a model with joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import joblib
from os import path

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")


def load_model(model_filename="neuronal_model.joblib"):
    model = joblib.load(path.join(output_path, model_filename))
    return model
