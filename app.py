from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
import os

# =========================
# FIX: Limit ONNX Runtime threads to prevent segfault
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNX_RUNTIME_EXECUTION_PROVIDER"] = "CPUExecutionProvider"

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

logging.basicConfig(level=logging.INFO)

# =========================
# CLASSES
# =========================
CLASSES = ['FMD', 'lumpy disease', 'mastitis', 'normal']

# =========================
# DISEASE INFO
# =========================
DISEASE_INFO = {
    'FMD': {
        'name': 'Foot-and-Mouth Disease (FMD)',
        'description': 'Highly contagious viral disease affecting cattle.',
        'urgency': 'Critical'
    },
    'lumpy disease': {
        'name': 'Lumpy Skin Disease',
        'description': 'Viral disease causing skin nodules in cattle.',
        'urgency': 'High'
    },
    'mastitis': {
        'name': 'Mastitis',
        'description': 'Udder infection causing milk production issues.',
        'urgency': 'Moderate'
    },
    'normal': {
        'name': 'Healthy Cow',
        'description': 'No disease detected.',
        'urgency': 'Normal'
    }
}

# =========================
# LOAD ONNX MODEL WITH THREAD LIMITS
# =========================
session = None
input_name = None

try:
    if not os.path.exists("cow_disease_model.onnx"):
        logging.error("❌ Model file 'cow_disease_model.onnx' not found!")
    else:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(
            "cow_disease_model.onnx",
            sess_options,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        logging.info("✅ ONNX Model loaded successfully")

except Exception as e:
    logging.error(f"❌ Model load failed: {e}")
    session = None

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    # Normalize to [-1, 1] (adjust if your model expects [0,1])
    img = (img - 0.5) / 0.5
    img = np.expand_dims(img, axis=0)   # batch
    
    return img

# =========================
# SOFTMAX
# =========================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image_bytes):
    if session is None:
        return {"error": "Model not loaded properly on the server"}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = preprocess_image(image)

        outputs = session.run(None, {input_name: img})[0]
        probs = softmax(outputs[0])

        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        probabilities = {
            cls: float(probs[i]) * 100 for i, cls in enumerate(CLASSES)
        }

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "disease_info": DISEASE_INFO[pred_class]
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}

# =========================
# ROUTES (WITH CACHE CONTROL)
# =========================
@app.route("/")
def home():
    # Render template and wrap in a response object
    resp = make_response(render_template("index.html"))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/manifest.json')
def serve_manifest():
    resp = make_response(send_from_directory('static', 'manifest.json'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

@app.route('/sw.js')
def serve_sw():
    resp = make_response(send_from_directory('static', 'sw.js'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

@app.route('/<path:filename>')
def serve_icons(filename):
    if filename.endswith('.png') or filename.endswith('.ico'):
        return send_from_directory('static', filename)
    return "File not found", 404

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image_bytes = file.read()
        result = predict_image(image_bytes)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if session else "unhealthy",
        "model_loaded": session is not None
    })

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)