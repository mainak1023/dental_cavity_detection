import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Resize image
    img = cv2.resize(img, img_size)
    
    # Normalize image
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(model, image_path, class_names, img_size=(224, 224)):
    """
    Make a prediction for a single image.
    
    Args:
        model: Trained model
        image_path (str): Path to the image file
        class_names (list): List of class names
        img_size (tuple): Target image size (height, width)
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Load and preprocess image
    img = load_and_preprocess_image(image_path, img_size)
    
    # Make prediction
    predictions = model.predict(img)[0]
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return predicted_class, confidence

def main(args):
    # Load model
    model = load_model(args.model_path)
    
    # Load class names
    with open(args.class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Process single image or directory
    if os.path.isfile(args.input_path):
        # Process single image
        predicted_class, confidence = predict_image(
            model, args.input_path, class_names, (args.img_size, args.img_size)
        )
        
        # Display result
        img = cv2.imread(args.input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Predicted: {class_names[predicted_class]} (Confidence: {confidence:.2f})')
        plt.axis('off')
        plt.show()
        
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}")
        
    elif os.path.isdir(args.input_path):
        # Process all images in directory
        image_files = [f for f in os.listdir(args.input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        plt.figure(figsize=(15, 10))
        
        for i, image_file in enumerate(image_files[:9]):  # Display up to 9 images
            image_path = os.path.join(args.input_path, image_file)
            
            # Make prediction
            predicted_class, confidence = predict_image(
                model, image_path, class_names, (args.img_size, args.img_size)
            )
            
            # Display image and prediction
            plt.subplot(3, 3, i + 1)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f'{class_names[predicted_class]} ({confidence:.2f})')
            plt.axis('off')
            
            print(f"{image_file}: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
        
        plt.tight_layout()
        plt.show()
    
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dental Cavity Detection Prediction')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--class_names_path', type=str, required=True,
                        help='Path to file containing class names (one per line)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (height and width)')
    
    args = parser.parse_args()
    main(args)

