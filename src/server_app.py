from flask import Flask, request, jsonify, send_from_directory
import os
from sklearn.feature_extraction.text import CountVectorizer
from load_model import load_model
from get_vocabulary import get_vocabulary, get_vocabulary_count
from flask_cors import CORS

app = Flask(__name__)

# Configurar la carpeta pública
public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../public")
os.makedirs(public_dir, exist_ok=True)
app.static_folder = public_dir

# Cargar el vocabulario y el modelo en el inicio del servidor
vocabulary = get_vocabulary()
model = load_model()

# Cors para permitir solicitudes de cualquier origen
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener el texto del cuerpo de la solicitud
        data = request.get_json()
        if not data or "text" not in data:
            return (
                jsonify(
                    {
                        "error": "El cuerpo de la solicitud debe contener un campo 'text'."
                    }
                ),
                400,
            )

        email = data["text"]
        print(email)

        # Vectorizar el texto usando el vocabulario
        counts = get_vocabulary_count(email, vocabulary)

        # Realizar la predicción
        prediction = model.predict([counts])[0]

        # Convertir la predicción a formato legible
        result = "spam" if prediction == 1 else "ham"

        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(public_dir, "index.html")


@app.route("/static/<path:filename>", methods=["GET"])
def serve_static(filename):
    return send_from_directory(public_dir, filename)


if __name__ == "__main__":
    index_html = os.path.join(public_dir, "index.html")
    app.run(host="0.0.0.0", port=5000, debug=True)
