import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

class DentalCavityModel:
    def __init__(self, input_shape, num_classes, model_type='custom'):
        """
        Initialize the dental cavity detection model.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of classes
            model_type (str): Type of model to use ('custom', 'vgg16', 'resnet50', 'mobilenetv2')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the model architecture based on the specified model type.
        
        Returns:
            tensorflow.keras.Model: The compiled model
        """
        if self.model_type == 'custom':
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                MaxPooling2D((2, 2)),
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        elif self.model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            base_model.trainable = False  # Freeze the base model
            
            inputs = Input(shape=self.input_shape)
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
        elif self.model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
            base_model.trainable = False  # Freeze the base model
            
            inputs = Input(shape=self.input_shape)
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
        elif self.model_type == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
            base_model.trainable = False  # Freeze the base model
            
            inputs = Input(shape=self.input_shape)
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs, outputs)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, save_dir='models'):
        """
        Train the model.
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation images
            y_val (numpy.ndarray): Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            save_dir (str): Directory to save model checkpoints
            
        Returns:
            dict: Training history
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(save_dir, f'dental_cavity_{self.model_type}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        
        # Save the final model
        self.model.save(os.path.join(save_dir, f'dental_cavity_{self.model_type}_final.h5'))
        
        return history.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        
        Args:
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X (numpy.ndarray): Input images
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self.model.predict(X)
    
    def plot_training_history(self, history, save_path='results/training_history.png'):
        """
        Plot the training history.
        
        Args:
            history (dict): Training history
            save_path (str): Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

