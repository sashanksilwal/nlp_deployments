from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, torch, json
from sarcasm_classifier import SarcasmClassifier

app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
CORS(app)

# --- load model ---
with open("vocab.json") as f:
    vocab = json.load(f)
model = SarcasmClassifier(vocab=vocab, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "missing text"}), 400
    pred = model.predict([text])
    return jsonify({"prediction": int(pred[0].item())})

# serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
