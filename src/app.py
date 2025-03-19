import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model
from utils import create_gradcam_visualization

def load_and_preprocess_image(image, img_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image: Input image (numpy array or file path)
        img_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Check if image is a file path or numpy array
    if isinstance(image, str):
        # Read image from file
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            img = image
    
    # Resize image
    img = cv2.resize(img, img_size)
    
    # Normalize image
    img_normalized = img / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img

def predict_image(model, image, class_names, img_size=(224, 224)):
    """
    Make a prediction for a single image.
    
    Args:
        model: Trained model
        image: Input image (numpy array or file path)
        class_names (list): List of class names
        img_size (tuple): Target image size (height, width)
        
    Returns:
        tuple: (predicted_class, confidence, preprocessed_image)
    """
    # Load and preprocess image
    img_batch, original_img = load_and_preprocess_image(image, img_size)
    
    # Make prediction
    predictions = model.predict(img_batch)[0]
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return predicted_class, confidence, img_batch, original_img

def dental_cavity_detection(image, model_path, class_names_path, visualize_gradcam=True):
    """
    Perform dental cavity detection on an input image.
    
    Args:
        image: Input image
        model_path (str): Path to the trained model
        class_names_path (str): Path to the file containing class names
        visualize_gradcam (bool): Whether to visualize Grad-CAM
        
    Returns:
        tuple: (result_image, prediction_text)
    """
    # Load model
    model = load_model(model_path)
    
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Make prediction
    predicted_class, confidence, img_batch, original_img = predict_image(
        model, image, class_names, (224, 224)
    )
    
    # Create result text
    prediction_text = f"Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2f}"
    
    # Create result visualization
    if visualize_gradcam:
        # Generate Grad-CAM visualization
        gradcam_img = create_gradcam_visualization(model, img_batch, predicted_class)
        
        # Create a figure with original and Grad-CAM images
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
        
        # Convert buffer to image
        result_image = Image.open(buf)
        plt.close()
    else:
        # Just add text to the original image
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(prediction_text)
        plt.axis('off')
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert buffer to image
        result_image = Image.open(buf)
        plt.close()
    
    return result_image, prediction_text

def create_gradio_interface():
    """
    Create a Gradio interface for dental cavity detection.
    
    Returns:
        gradio.Interface: Gradio interface
    """
    # Define the interface
    iface = gr.Interface(
        fn=lambda img: dental_cavity_detection(
            img, 
            model_path='models/dental_cavity_mobilenetv2_finetuned_best.h5',
            class_names_path='models/class_names.txt'
        ),
        inputs=gr.Image(),
        outputs=[
            gr.Image(label="Result Visualization"),
            gr.Textbox(label="Prediction")
        ],
        title="Dental Cavity Detection",
        description="Upload a dental image to detect cavities and other dental conditions.",
        examples=[
            ["examples/cavity_example1.jpg"],
            ["examples/cavity_example2.jpg"],
            ["examples/normal_example1.jpg"]
        ]
    )
    
    return iface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dental Cavity Detection App')
    
    parser.add_argument('--mode', type=str, default='gradio',
                        choices=['gradio', 'cli'],
                        help='Mode to run the app (gradio or cli)')
    parser.add_argument('--model_path', type=str, 
                        default='models/dental_cavity_mobilenetv2_finetuned_best.h5',
                        help='Path to the trained model file')
    parser.add_argument('--class_names_path', type=str, 
                        default='models/class_names.txt',
                        help='Path to file containing class names (one per line)')
    parser.add_argument('--input_path', type=str,
                        help='Path to input image (required for cli mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'gradio':
        # Create and launch Gradio interface
        iface = create_gradio_interface()
        iface.launch()
    
    elif args.mode == 'cli':
        # Check if input path is provided
        if not args.input_path:
            print("Error: --input_path is required for cli mode")
            parser.print_help()
            exit(1)
        
        # Check if input path exists
        if not os.path.exists(args.input_path):
            print(f"Error: Input path {args.input_path} does not exist")
            exit(1)
        
        # Load model
        model = load_model(args.model_path)
        
        # Load class names
        with open(args.class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Make prediction
        predicted_class, confidence, _, _ = predict_image(
            model, args.input_path, class_names, (224, 224)
        )
        
        # Print result
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}")

