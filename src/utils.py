import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import seaborn as sns

def save_class_names(class_names, save_path='models/class_names.txt'):
    """
    Save class names to a file.
    
    Args:
        class_names (list): List of class names
        save_path (str): Path to save the class names
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Write class names to file
    with open(save_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

def load_class_names(file_path='models/class_names.txt'):
    """
    Load class names from a file.
    
    Args:
        file_path (str): Path to the file containing class names
        
    Returns:
        list: List of class names
    """
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names

def create_gradcam_visualization(model, img_array, class_idx, layer_name=None):
    """
    Create a Grad-CAM visualization for the given image.
    
    Args:
        model: Trained model
        img_array (numpy.ndarray): Input image (should be preprocessed)
        class_idx (int): Index of the class to visualize
        layer_name (str): Name of the layer to use for Grad-CAM
        
    Returns:
        numpy.ndarray: Grad-CAM visualization
    """
    import tensorflow as tf
    
    # If layer_name is not provided, use the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    # Get the gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute guided gradients
    cast_conv_outputs = tf.cast(conv_outputs > 0, tf.float32)
    cast_grads = tf.cast(grads > 0, tf.float32)
    guided_grads = cast_conv_outputs * cast_grads * grads
    
    # Global average pooling
    weights = tf.reduce_mean(guided_grads, axis=(1, 2))
    
    # Create class activation map
    cam = np.ones(conv_outputs.shape[1:3], dtype=np.float32)
    
    for i, w in enumerate(weights[0]):
        cam += w * conv_outputs[0, :, :, i].numpy()
    
    # Apply ReLU to the CAM
    cam = np.maximum(cam, 0)
    
    # Normalize the CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
    
    # Resize to the input image size
    cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[2]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    img = (img_array[0] * 255).astype(np.uint8)
    superimposed = heatmap * 0.4 + img
    superimposed = superimposed / superimposed.max() * 255
    
    return superimposed.astype(np.uint8)

def visualize_model_layers(model, img, layer_indices=None, save_path='results/layer_visualization.png'):
    """
    Visualize the activations of different layers in the model.
    
    Args:
        model: Trained model
        img (numpy.ndarray): Input image (should be preprocessed and have batch dimension)
        layer_indices (list): Indices of layers to visualize
        save_path (str): Path to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # If layer_indices is not provided, select convolutional layers
    if layer_indices is None:
        layer_indices = []
        for i, layer in enumerate(model.layers):
            if 'conv' in layer.name:
                layer_indices.append(i)
    
    # Create a model that will return the activations for the specified layers
    layer_outputs = [model.layers[idx].output for idx in layer_indices]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(img)
    
    # Plot activations
    plt.figure(figsize=(15, 10))
    
    for i, activation in enumerate(activations):
        layer_name = model.layers[layer_indices[i]].name
        
        # For convolutional layers, plot a subset of feature maps
        n_features = min(16, activation.shape[-1])
        size = int(np.ceil(np.sqrt(n_features)))
        
        for j in range(n_features):
            plt.subplot(len(activations), size, i * size + j + 1)
            plt.imshow(activation[0, :, :, j], cmap='viridis')
            plt.title(f"{layer_name}\nFeature {j}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_learning_curves(history, metrics=['accuracy', 'loss'], save_path='results/learning_curves.png'):
    """
    Plot learning curves for the specified metrics.
    
    Args:
        history (dict): Training history
        metrics (list): List of metrics to plot
        save_path (str): Path to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 5 * len(metrics)))
    
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        
        plt.plot(history[metric], label=f'Training {metric}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
        
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

