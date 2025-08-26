from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import io
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("Model_Save/final_efficientnet_b0_cv.pth")
CLASS_NAMES = ["LG", "HG"]

# Load model
def load_model():
    try:
        model = models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

model = load_model()

# Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading HTML: {e}"

@app.route('/test')
def test():
    return jsonify({'status': 'ok'})

@app.route('/docs')
def docs():
    try:
        with open('docs.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading documentation: {e}"

@app.route('/results')
def results():
    try:
        with open('results.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading results: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        file = request.files['image']
        image_bytes = file.read()
        
        # Preprocess and predict
        image_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100
        class_name = CLASS_NAMES[predicted_class]
        
        return jsonify({
            'prediction': class_name,
            'confidence': round(confidence, 2),
            'probabilities': {
                'LG': round(probs[0][0].item() * 100, 2),
                'HG': round(probs[0][1].item() * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)