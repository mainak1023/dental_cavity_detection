import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import itertools

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained model
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels (one-hot encoded)
            class_names (list): List of class names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_pred_proba = model.predict(X_test)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        self.y_true = np.argmax(y_test, axis=1)
        
    def plot_confusion_matrix(self, normalize=False, save_path='results/confusion_matrix.png'):
        """
        Plot the confusion matrix.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            save_path (str): Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
    def print_classification_report(self):
        """
        Print the classification report.
        """
        report = classification_report(self.y_true, self.y_pred, 
                                      target_names=self.class_names)
        print("Classification Report:")
        print(report)
        
        # Save report to file
        with open('results/classification_report.txt', 'w') as f:
            f.write(report)
        
    def plot_roc_curve(self, save_path='results/roc_curve.png'):
        """
        Plot the ROC curve for each class.
        
        Args:
            save_path (str): Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot ROC curve for each class
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            # Get true labels and predicted probabilities for this class
            y_true_class = (self.y_true == i).astype(int)
            y_pred_proba_class = self.y_pred_proba[:, i]
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true_class, y_pred_proba_class)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        
    def visualize_predictions(self, num_samples=5, save_path='results/predictions.png'):
        """
        Visualize model predictions on random test samples.
        
        Args:
            num_samples (int): Number of samples to visualize per class
            save_path (str): Path to save the plot
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get indices of images belonging to this class
            indices = np.where(self.y_true == class_idx)[0]
            
            # Select random samples
            if len(indices) >= num_samples:
                samples = np.random.choice(indices, num_samples, replace=False)
            else:
                samples = indices
            
            # Plot samples
            for i, sample_idx in enumerate(samples):
                plt.subplot(len(self.class_names), num_samples, class_idx * num_samples + i + 1)
                plt.imshow(self.X_test[sample_idx])
                
                # Get predicted class and probability
                pred_class = self.y_pred[sample_idx]
                pred_prob = self.y_pred_proba[sample_idx, pred_class]
                
                # Set title color based on prediction correctness
                title_color = 'green' if pred_class == class_idx else 'red'
                
                plt.title(f'True: {class_name}\nPred: {self.class_names[pred_class]} ({pred_prob:.2f})', 
                          color=title_color)
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

