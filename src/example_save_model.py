# This is only expample of how to save a model with joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import joblib
from os import path

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")


# Crear datos de ejemplo
def create_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Entrenar y guardar el modelo
def train_and_save_model(X_train, y_train):
    # Crear una red neuronal sencilla
    model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(
        model, path.join(output_path, "iris_model.joblib")
    )  # Guardar el modelo en un archivo
    print(f"Modelo guardado como {path.join(output_path, 'iris_model.joblib')}")


def load_model():
    model = joblib.load(path.join(output_path, "iris_model.joblib"))
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_iris_data()

    train_and_save_model(X_train, y_train)

    model = joblib.load(path.join(output_path, "iris_model.joblib"))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
