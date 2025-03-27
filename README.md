Web-based Application for Denoising Surveillance Camera Footage using RedNet
About the Project
This project presents a web-based application that intelligently denoises surveillance camera footage using deep learning. It integrates a Noise Classifier Model that identifies the type of noise present in an image and applies a corresponding RedNet-based denoising model to enhance image clarity.

Key Features
✅ Noise Classification: Detects four types of noise in images:

Low Light

Compression Artifacts

Rain/Snow Distortion

Motion Blur

✅ Adaptive Denoising: Based on the detected noise type, the system applies a trained RedNet model to remove noise effectively.

✅ Web-Based Interface: Users can upload noisy images through a simple web application, and the system returns high-quality, denoised images.

Technologies Used
Deep Learning: RedNet for denoising, CNN-based classifier for noise detection

Frontend: HTML, CSS, JavaScript

Backend: Flask

Model Training: PyTorch 

How It Works
User uploads an image via the web interface.

The noise classifier predicts the type of noise present in the image.

The appropriate RedNet model is applied to denoise the image.

The enhanced image is returned to the user for download.
