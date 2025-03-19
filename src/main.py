import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from model import DentalCavityModel
from evaluation import ModelEvaluator

def main(args):
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = preprocessor.load_and_preprocess_data()
    
    # Visualize sample images
    preprocessor.visualize_samples(X_train, y_train, class_names)
    
    # Apply data augmentation if enabled
    if args.augment:
        X_train, y_train = preprocessor.apply_augmentation(X_train, y_train)
    
    # Initialize model
    input_shape = X_train.shape[1:]  # (height, width, channels)
    num_classes = len(class_names)
    
    model = DentalCavityModel(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type=args.model_type
    )
    
    # Print model summary
    model.model.summary()
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Plot training history
    model.plot_training_history(history)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Detailed evaluation
    evaluator = ModelEvaluator(model.model, X_test, y_test, class_names)
    evaluator.plot_confusion_matrix()
    evaluator.print_classification_report()
    evaluator.plot_roc_curve()
    evaluator.visualize_predictions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dental Cavity Detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/cavity_dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (height and width)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='mobilenetv2',
                        choices=['custom', 'vgg16', 'resnet50', 'mobilenetv2'],
                        help='Type of model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    main(args)

