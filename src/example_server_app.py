from flask import Flask, request, jsonify
import joblib
import numpy as np
from os import path

app = Flask(__name__)

current_dir = path.dirname(path.abspath(__file__))
output_path = path.join(current_dir, "../output")
# Cargar el modelo
model = joblib.load(path.join(output_path, "iris_model.joblib"))


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Obtener los parámetros de la solicitud
        feature_1 = float(request.args.get("feature_1"))
        feature_2 = float(request.args.get("feature_2"))
        feature_3 = float(request.args.get("feature_3"))
        feature_4 = float(request.args.get("feature_4"))

        # Crear el array con las características
        features = np.array([[feature_1, feature_2, feature_3, feature_4]])

        # Hacer la predicción
        prediction = model.predict(features)

        # Responder con la predicción
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
