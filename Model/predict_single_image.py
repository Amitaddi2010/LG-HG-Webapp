import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import argparse
from pathlib import Path
import cv2
import numpy as np

# --- Configuration ---
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("C:/ESS_AI_Project/models/final_efficientnet_b0_cv.pth")
CLASS_NAMES = ["LG", "HG"]  # Class 0: LG, Class 1: HG

# --- Load the Model ---
def load_model(model_path):
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# --- Preprocess Image (Matching transform_images224.py) ---
def preprocess_image(img_path):
    # Read image with OpenCV (BGR format)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img)
    
    # Apply the same transform sequence as in preprocessing
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess and return tensor
    image_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    return image_tensor

def predict_image(img_path, model):
    # Preprocess the image
    image_tensor = preprocess_image(img_path)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    
    # Convert to class name and confidence percentage
    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100
    class_name = CLASS_NAMES[predicted_class]

    return class_name, confidence

if __name__ == "__main__":
    # Parse command line argument for image path
    parser = argparse.ArgumentParser(description="Predict class (HG vs LG) for a single image.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image file to predict.")
    args = parser.parse_args()

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    try:
        class_name, confidence = predict_image(Path(args.img_path), model)
        print(f"Prediction: {class_name} (Confidence: {confidence:.2f}%)")
    except Exception as e:
        print(f"Error: {str(e)}")