from flask import Flask, request, jsonify, render_template
import os, uuid, tempfile
from src.inference import load_model, predict_file
app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
model = None
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        # lazy load (if model not present, reply with helpful message)
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error':'Model not found. Train the model first (run src/train.py) to create best_model.pth'}), 400
        model = load_model(MODEL_PATH)
    if 'file' not in request.files:
        return jsonify({'error':'no file uploaded'}), 400
    f = request.files['file']
    fname = str(uuid.uuid4()) + '_' + f.filename
    tmp = os.path.join(tempfile.gettempdir(), fname)
    f.save(tmp)
    try:
        res = predict_file(model, tmp)
    finally:
        try:
            os.remove(tmp)
        except:
            pass
    return jsonify(res)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
