from feature_extractor import FeatureExtractor
from image_processor import ImageProcessor
from svm_models import SVM_RBF
from joblib import load  # To load your SVM model
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Path to saved models
MODEL_RBF_PATH = "models_rbf/svm_rbf_model_C_10_gamma_0.001.joblib"

# Load SVM model (joblib format)
model_rbf = load(MODEL_RBF_PATH)

# Initialize FeatureExtractor and ImageProcessor
feature_extractor = FeatureExtractor()
image_processor = ImageProcessor()

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page for GUI

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save uploaded file to a temporary location
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        try:
            # Preprocess and predict
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = np.array(img)
            roi = image_processor.extract_roi(img_array)
            processed_img = image_processor.resize_and_pad(roi)
            normalized_img = image_processor.normalize(processed_img)
            features = feature_extractor.get_features(normalized_img)
            rbf_prediction = model_rbf.predict([features])

            os.remove(filepath)  # Clean up uploaded file
            return jsonify({"rbf_prediction": int(rbf_prediction[0])})

        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
