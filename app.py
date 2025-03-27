from flask import Flask, request, send_from_directory, render_template, jsonify
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
import traceback

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")

# Device Configuration (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define RedNet Model (Denoiser)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class RedNet(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(RedNet, self).__init__()
        self.input_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.output_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.output_layer(out)
        return out

# Load Noise Classification Model
models_dir = r"C:\Users\keshi\OneDrive\Documents\projects\denoising_web\models"
try:
    noise_classifier_path = os.path.join(models_dir, "noise_classifier.h5")
    noise_classifier = tf.keras.models.load_model(noise_classifier_path)
    print(f"Noise classifier loaded from: {noise_classifier_path}")
except Exception as e:
    print(f"Error loading noise classifier: {e}")
    exit()

# Load Single Denoising Model (Gaussian Model)
denoising_model_path = os.path.join(models_dir, "gaussian_model.pth")
try:
    denoising_model = RedNet()
    denoising_model.load_state_dict(torch.load(denoising_model_path, map_location=device))
    denoising_model.eval()
    denoising_model.to(device)
    print(f"Single denoising model (Gaussian) loaded from: {denoising_model_path}")
except Exception as e:
    print(f"Error loading Gaussian denoising model: {e}")
    exit()

# File Upload Configurations
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_image(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)
    return file_path

# Serve Frontend (HTML, CSS, JS)
@app.route('/')
def serve_frontend():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Image Preprocessing for TensorFlow Model
def preprocess_image_tf(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Noise Types
noise_types = ["compression", "low_light", "motion_blur", "rain_fog", "speckle"]

# Denoising API Endpoint
@app.route('/denoise', methods=['POST'])
def denoise_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if uploaded_file and allowed_file(uploaded_file.filename):
        image_path = save_uploaded_image(uploaded_file)

        try:
            # Step 1: Predict Noise Type
            image_array = preprocess_image_tf(image_path)
            predicted_noise_probs = noise_classifier.predict(image_array)
            predicted_class_index = np.argmax(predicted_noise_probs, axis=-1)[0]

            if 0 <= predicted_class_index < len(noise_types):
                predicted_noise_type = noise_types[predicted_class_index]
            else:
                predicted_noise_type = "Unknown Noise Type"

            print(f"Predicted Noise Type: {predicted_noise_type}")  # PRINT in Command Window

            # Step 2: Apply Single Denoising Model (Gaussian)
            noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            noisy_image_for_denoising = cv2.resize(noisy_image, (128, 128))
            noisy_image_for_denoising = noisy_image_for_denoising.astype(np.float32) / 255.0
            noisy_image_tensor = torch.from_numpy(noisy_image_for_denoising).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                denoised_image_tensor = denoising_model(noisy_image_tensor)

            # Convert Tensor to Image
            denoised_image = denoised_image_tensor.cpu().squeeze().numpy()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            denoised_filename = f"denoised_{timestamp}.png"
            denoised_image_path = os.path.join(UPLOAD_FOLDER, denoised_filename)

            denoised_image = (denoised_image * 255).astype(np.uint8)
            if not cv2.imwrite(denoised_image_path, denoised_image):
                return jsonify({"error": "Failed to save denoised image"}), 500

            return jsonify({
                "output_image_url": f"/uploads/{denoised_filename}",
                "noise_type": predicted_noise_type,
                "noisy_image_url": f"/uploads/{uploaded_file.filename}"
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
