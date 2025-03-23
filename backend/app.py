import os
import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import create_gradcam_visualization

app = FastAPI(title="Dental Cavity Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class names
model = None
class_names = []

@app.on_event("startup")
async def startup_event():
    """Load model and class names on startup"""
    global model, class_names
    
    # Load model
    model_path = os.path.join('models', 'dental_cavity_mobilenetv2_finetuned_best.h5')
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print(f"Warning: Model file {model_path} not found. API will not work until model is available.")
    
    # Load class names
    class_names_path = os.path.join('models', 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print(f"Warning: Class names file {class_names_path} not found.")

def preprocess_image(image_bytes):
    """
    Preprocess image from bytes.
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        tuple: (preprocessed_image_batch, original_image)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img, (224, 224))
    
    # Normalize image
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img

def encode_image_to_base64(image):
    """
    Encode image to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        str: Base64 encoded image
    """
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype('uint8'))
    else:
        image_pil = image
    
    # Save image to bytes buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    
    # Encode bytes to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict dental cavity from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        dict: Prediction results
    """
    global model, class_names
    
    # Check if model and class names are loaded
    if model is None or not class_names:
        return JSONResponse(
            status_code=503,
            content={"error": "Model or class names not loaded. Please try again later."}
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Preprocess image
        img_batch, original_img = preprocess_image(contents)
        
        # Make prediction
        predictions = model.predict(img_batch)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        # Generate Grad-CAM visualization
        gradcam_img = create_gradcam_visualization(model, img_batch, predicted_class)
        
        # Create result images
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_img)
        plt.title(f"Grad-CAM: {class_names[predicted_class]} ({confidence:.2f})")
        plt.axis('off')
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convert buffer to image
        result_image = Image.open(buf)
        
        # Encode images to base64
        original_img_base64 = encode_image_to_base64(original_img)
        gradcam_img_base64 = encode_image_to_base64(gradcam_img)
        result_img_base64 = encode_image_to_base64(result_image)
        
        # Return results
        return {
            "class": class_names[predicted_class],
            "confidence": confidence,
            "original_image": original_img_base64,
            "gradcam_image": gradcam_img_base64,
            "result_image": result_img_base64
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/classes")
async def get_classes():
    """
    Get available class names.
    
    Returns:
        dict: Class names
    """
    global class_names
    
    return {"classes": class_names}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    global model, class_names
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) > 0,
        "num_classes": len(class_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)