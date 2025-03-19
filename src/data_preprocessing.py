import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(224, 224), test_size=0.2, val_size=0.1):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory containing the dataset
            img_size (tuple): Target image size (height, width)
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size
        self.classes = None
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dental cavity images.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        # Get all subdirectories (class folders)
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
                
            class_names.append(class_dir)
            class_idx = len(class_names) - 1
            
            # Process images in each class directory
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                # Skip if not an image file
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize to [0,1]
                    
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # One-hot encode the labels
        y_categorical = to_categorical(y, num_classes=len(class_names))
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_categorical, test_size=self.test_size, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size/(1-self.test_size), 
            random_state=42
        )
        
        self.classes = class_names
        
        print(f"Dataset loaded: {len(X)} images")
        print(f"Classes: {class_names}")
        print(f"Train set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names
    
    def visualize_samples(self, X, y, class_names, num_samples=5):
        """
        Visualize sample images from the dataset.
        
        Args:
            X (numpy.ndarray): Image data
            y (numpy.ndarray): One-hot encoded labels
            class_names (list): List of class names
            num_samples (int): Number of samples to visualize per class
        """
        # Convert one-hot encoded labels back to indices
        y_indices = np.argmax(y, axis=1)
        
        plt.figure(figsize=(15, 10))
        
        for class_idx, class_name in enumerate(class_names):
            # Get indices of images belonging to this class
            indices = np.where(y_indices == class_idx)[0]
            
            # Select random samples
            if len(indices) >= num_samples:
                samples = np.random.choice(indices, num_samples, replace=False)
            else:
                samples = indices
            
            # Plot samples
            for i, sample_idx in enumerate(samples):
                plt.subplot(len(class_names), num_samples, class_idx * num_samples + i + 1)
                plt.imshow(X[sample_idx])
                plt.title(class_name)
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_images.png')
        plt.show()
    
    def apply_augmentation(self, X_train, y_train):
        """
        Apply data augmentation to the training set.
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels
            
        Returns:
            tuple: (augmented_images, augmented_labels)
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Create data generator with augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Generate augmented data
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(X_train)):
            img = X_train[i].reshape(1, *X_train[i].shape)
            label = y_train[i].reshape(1, -1)
            
            # Generate 5 augmented versions of each image
            for batch in datagen.flow(img, label, batch_size=1):
                augmented_images.append(batch[0][0])
                augmented_labels.append(batch[1][0])
                
                if len(augmented_images) % 5 == 0:
                    break
        
        # Combine original and augmented data
        X_augmented = np.vstack([X_train, np.array(augmented_images)])
        y_augmented = np.vstack([y_train, np.array(augmented_labels)])
        
        print(f"Original training set: {X_train.shape[0]} images")
        print(f"Augmented training set: {X_augmented.shape[0]} images")
        
        return X_augmented, y_augmented

