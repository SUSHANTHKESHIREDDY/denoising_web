from flask import Flask, request, send_file, send_from_directory
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static")

# Load the trained model (RedNet)
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
        out = self.input_layer(x.cuda())
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.output_layer(out)
        return out

# Load model
model = RedNet(num_residual_blocks=5)
model.load_state_dict(torch.load('C:/Users/keshi/Downloads/rednet_model_All.pth'))
model.eval()

# Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload folder
UPLOAD_FOLDER = "C:/Users/keshi/Downloads/res"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to save uploaded file
def save_uploaded_image(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)
    return file_path

# ✅ Serve Frontend (HTML, CSS, JS)
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Route for image denoising
@app.route('/denoise', methods=['POST'])
def denoise_image():
    if 'image' not in request.files:
        return "No file part", 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return "No selected file", 400

    if uploaded_file and allowed_file(uploaded_file.filename):
        image_path = save_uploaded_image(uploaded_file)

        # Load and preprocess the noisy image
        noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        noisy_image = cv2.resize(noisy_image, (128, 128)).astype(np.float32) / 255.0

        # Convert to PyTorch tensor and add batch dimension
        noisy_image_tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            denoised_image_tensor = model(noisy_image_tensor)

        # Convert the denoised image back to a NumPy array
        denoised_image = denoised_image_tensor.cpu().squeeze().numpy()

        # Save denoised image
        denoised_image_path = os.path.join(UPLOAD_FOLDER, 'denoised_image.png')
        denoised_image = (denoised_image * 255).astype(np.uint8)
        cv2.imwrite(denoised_image_path, denoised_image)

        # Return denoised image as a downloadable file
        return send_file(denoised_image_path, as_attachment=True)

    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
