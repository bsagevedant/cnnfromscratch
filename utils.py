"""
Bhaskar's CNN Utilities Module
=============================
Utility functions for data processing, model analysis, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import json

def visualize_data_distribution(y_train, y_val=None, y_test=None, title="Data Distribution"):
    """Visualize the distribution of classes in the dataset"""
    fig, axes = plt.subplots(1, 2 if y_val is not None else 1, figsize=(15, 5))
    
    if y_val is not None:
        axes = [axes] if not hasattr(axes, '__len__') else axes
    
    # Training data
    train_classes = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    unique, counts = np.unique(train_classes, return_counts=True)
    
    axes[0].bar(unique, counts, alpha=0.7, color='skyblue')
    axes[0].set_title(f'{title} - Training Set')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Validation data
    if y_val is not None and len(axes) > 1:
        val_classes = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
        unique_val, counts_val = np.unique(val_classes, return_counts=True)
        
        axes[1].bar(unique_val, counts_val, alpha=0.7, color='lightcoral')
        axes[1].set_title(f'{title} - Validation Set')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title='Confusion Matrix'):
    """Plot confusion matrix with optional normalization"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm

def analyze_model_predictions(network, x_test, y_test, num_samples=20):
    """Analyze model predictions with detailed statistics"""
    predictions = []
    true_labels = []
    confidences = []
    
    for i in range(min(num_samples, len(x_test))):
        output = network.forward(x_test[i]) if hasattr(network, 'forward') else predict(network, x_test[i])
        pred = np.argmax(output)
        confidence = np.max(output)
        
        predictions.append(pred)
        true_labels.append(np.argmax(y_test[i]))
        confidences.append(confidence)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Print detailed analysis
    print(f"📊 Model Analysis Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Average Confidence: {np.mean(confidences):.4f}")
    print(f"   Confidence Std: {np.std(confidences):.4f}")
    
    # Show misclassified examples
    misclassified = predictions != true_labels
    if np.any(misclassified):
        print(f"   Misclassified: {np.sum(misclassified)}/{len(predictions)}")
        print(f"   Misclassified Confidence: {np.mean(confidences[misclassified]):.4f}")
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'confidences': confidences,
        'accuracy': accuracy,
        'misclassified': misclassified
    }

def visualize_feature_maps(network, input_image, layer_indices=None, max_features=16):
    """Visualize feature maps from convolutional layers"""
    if layer_indices is None:
        layer_indices = [0, 2, 4]  # First few conv layers
    
    feature_maps = []
    current_input = input_image
    
    for i, layer in enumerate(network):
        current_input = layer.forward(current_input)
        
        if i in layer_indices and hasattr(layer, 'output'):
            feature_maps.append((i, layer.output))
    
    # Plot feature maps
    for layer_idx, feature_map in feature_maps:
        if len(feature_map.shape) == 3:  # (channels, height, width)
            num_features = min(feature_map.shape[0], max_features)
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(min(16, num_features)):
                axes[i].imshow(feature_map[i], cmap='viridis')
                axes[i].set_title(f'Feature {i}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_features, 16):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps - Layer {layer_idx}')
            plt.tight_layout()
            plt.show()

def compare_architectures(results_dict, metric='accuracy'):
    """Compare different architecture performance"""
    architectures = list(results_dict.keys())
    metrics = [results_dict[arch][metric] for arch in architectures]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(architectures, metrics, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    
    # Add value labels on bars
    for bar, metric_value in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{metric_value:.4f}', ha='center', va='bottom')
    
    plt.title(f'Architecture Comparison - {metric.title()}')
    plt.ylabel(metric.title())
    plt.xlabel('Architecture')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_experiment_results(results, filepath):
    """Save experiment results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_results[key][sub_key] = sub_value.tolist()
                else:
                    serializable_results[key][sub_key] = sub_value
        else:
            serializable_results[key] = value
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Experiment results saved to {filepath}")

def load_experiment_results(filepath):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"📂 Experiment results loaded from {filepath}")
    return results

def plot_training_comparison(histories, labels, metric='loss'):
    """Compare training histories of different models"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for history, label in zip(histories, labels):
        if hasattr(history, 'train_losses'):
            plt.plot(history.train_losses, label=f'{label} - Train', linestyle='-')
            if any(loss is not None for loss in history.val_losses):
                plt.plot(history.val_losses, label=f'{label} - Val', linestyle='--')
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for history, label in zip(histories, labels):
        if hasattr(history, 'train_accuracies'):
            if any(acc is not None for acc in history.train_accuracies):
                plt.plot(history.train_accuracies, label=f'{label} - Train', linestyle='-')
            if any(acc is not None for acc in history.val_accuracies):
                plt.plot(history.val_accuracies, label=f'{label} - Val', linestyle='--')
    
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_model_complexity(network):
    """Calculate model complexity metrics"""
    total_params = 0
    trainable_params = 0
    
    layer_info = []
    
    for i, layer in enumerate(network):
        layer_params = 0
        layer_trainable = 0
        
        if hasattr(layer, 'weights'):
            layer_params += layer.weights.size
            layer_trainable += layer.weights.size
        
        if hasattr(layer, 'kernels'):
            layer_params += layer.kernels.size
            layer_trainable += layer.kernels.size
        
        if hasattr(layer, 'bias'):
            layer_params += layer.bias.size
            layer_trainable += layer.bias.size
        
        if hasattr(layer, 'biases'):
            layer_params += layer.biases.size
            layer_trainable += layer.biases.size
        
        if hasattr(layer, 'gamma'):
            layer_params += layer.gamma.size
            layer_trainable += layer.gamma.size
        
        if hasattr(layer, 'beta'):
            layer_params += layer.beta.size
            layer_trainable += layer.beta.size
        
        total_params += layer_params
        trainable_params += layer_trainable
        
        layer_info.append({
            'layer': i,
            'type': type(layer).__name__,
            'parameters': layer_params,
            'trainable': layer_trainable
        })
    
    print(f"📊 Model Complexity Analysis:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    
    print(f"\n📋 Layer-wise Parameter Count:")
    for info in layer_info:
        if info['parameters'] > 0:
            print(f"   Layer {info['layer']} ({info['type']}): {info['parameters']:,} params")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_info': layer_info
    }

def create_learning_curves_animation(history, save_path=None):
    """Create animated learning curves (requires matplotlib animation)"""
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("⚠️  matplotlib animation not available. Install with: pip install matplotlib[animation]")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot loss
        ax1.plot(history.train_losses[:frame+1], 'b-', label='Train Loss')
        if any(loss is not None for loss in history.val_losses[:frame+1]):
            ax1.plot(history.val_losses[:frame+1], 'r-', label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if any(acc is not None for acc in history.train_accuracies[:frame+1]):
            ax2.plot(history.train_accuracies[:frame+1], 'b-', label='Train Acc')
        if any(acc is not None for acc in history.val_accuracies[:frame+1]):
            ax2.plot(history.val_accuracies[:frame+1], 'r-', label='Val Acc')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    anim = FuncAnimation(fig, animate, frames=len(history.train_losses), 
                        interval=200, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"🎬 Animation saved to {save_path}")
    
    plt.show()
    return anim

# Import predict function for compatibility
try:
    from network import predict
except ImportError:
    def predict(network, input, training=False):
        """Fallback predict function"""
        output = input
        for layer in network:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
            output = layer.forward(output)
        return output
