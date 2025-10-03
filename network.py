"""
Bhaskar's Advanced Neural Network Training Framework
==================================================
Sophisticated training framework with modern features:
- Batch processing
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training visualization
- Advanced optimizers
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Callable, Dict, Any
import matplotlib.pyplot as plt
import os
import json

class TrainingHistory:
    """Training history tracker with visualization capabilities"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss=None, train_acc=None, val_acc=None, lr=None):
        """Update training history"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', color='blue')
        if any(loss is not None for loss in self.val_losses):
            axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        if any(acc is not None for acc in self.train_accuracies):
            axes[0, 1].plot(self.epochs, self.train_accuracies, label='Train Acc', color='blue')
            if any(acc is not None for acc in self.val_accuracies):
                axes[0, 1].plot(self.epochs, self.val_accuracies, label='Val Acc', color='red')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate curve
        if any(lr is not None for lr in self.learning_rates):
            axes[1, 0].plot(self.epochs, self.learning_rates, color='green')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Combined view
        ax2 = axes[1, 1]
        ax2.plot(self.epochs, self.train_losses, label='Train Loss', color='blue')
        if any(loss is not None for loss in self.val_losses):
            ax2.plot(self.epochs, self.val_losses, label='Val Loss', color='red')
        ax2.set_title('Combined View')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, network):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = self._save_weights(network)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                self._restore_weights(network, self.best_weights)
            return True
        return False
    
    def _save_weights(self, network):
        """Save network weights"""
        weights = []
        for layer in network:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.copy())
            if hasattr(layer, 'kernels'):
                weights.append(layer.kernels.copy())
        return weights
    
    def _restore_weights(self, network, weights):
        """Restore network weights"""
        weight_idx = 0
        for layer in network:
            if hasattr(layer, 'weights'):
                layer.weights = weights[weight_idx].copy()
                weight_idx += 1
            if hasattr(layer, 'kernels'):
                layer.kernels = weights[weight_idx].copy()
                weight_idx += 1

class LearningRateScheduler:
    """Learning rate scheduling utilities"""
    
    @staticmethod
    def step_decay(initial_lr, decay_factor=0.5, step_size=10):
        """Step decay scheduler"""
        def scheduler(epoch):
            return initial_lr * (decay_factor ** (epoch // step_size))
        return scheduler
    
    @staticmethod
    def exponential_decay(initial_lr, decay_rate=0.96):
        """Exponential decay scheduler"""
        def scheduler(epoch):
            return initial_lr * (decay_rate ** epoch)
        return scheduler
    
    @staticmethod
    def cosine_annealing(initial_lr, max_epochs):
        """Cosine annealing scheduler"""
        def scheduler(epoch):
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        return scheduler

def predict(network, input, training=False):
    """Forward pass through network"""
    output = input
    for layer in network:
        if hasattr(layer, 'set_training'):
            layer.set_training(training)
        output = layer.forward(output)
    return output

def train_batch(network, batch_x, batch_y, loss, loss_prime, learning_rate, optimizer='sgd', **optimizer_kwargs):
    """Train on a single batch"""
    batch_loss = 0
    batch_size = len(batch_x)
    
    for x, y in zip(batch_x, batch_y):
        # Forward pass
        output = predict(network, x, training=True)
        
        # Compute loss
        batch_loss += loss(y, output)
        
        # Backward pass
        grad = loss_prime(y, output)
        for layer in reversed(network):
            if hasattr(layer, 'backward'):
                if optimizer == 'adam':
                    grad = layer.backward(grad, learning_rate, optimizer='adam', **optimizer_kwargs)
                elif optimizer == 'rmsprop':
                    grad = layer.backward(grad, learning_rate, optimizer='rmsprop', **optimizer_kwargs)
                else:
                    grad = layer.backward(grad, learning_rate, **optimizer_kwargs)
    
    return batch_loss / batch_size

def evaluate(network, x_val, y_val, loss, accuracy_fn=None):
    """Evaluate network on validation set"""
    val_loss = 0
    val_acc = 0
    correct = 0
    total = 0
    
    for x, y in zip(x_val, y_val):
        output = predict(network, x, training=False)
        val_loss += loss(y, output)
        
        if accuracy_fn:
            pred = np.argmax(output)
            true = np.argmax(y)
            if pred == true:
                correct += 1
            total += 1
    
    val_loss /= len(x_val)
    val_acc = correct / total if total > 0 else 0
    
    return val_loss, val_acc

def train(network, loss, loss_prime, x_train, y_train, 
          x_val=None, y_val=None, epochs=1000, learning_rate=0.01, 
          batch_size=32, optimizer='sgd', scheduler=None, early_stopping=None,
          verbose=True, save_path=None, **optimizer_kwargs):
    """
    Advanced training function with modern features
    
    Args:
        network: List of layers
        loss: Loss function
        loss_prime: Loss function derivative
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training
        optimizer: Optimizer type ('sgd', 'adam', 'rmsprop')
        scheduler: Learning rate scheduler function
        early_stopping: Early stopping object
        verbose: Print training progress
        save_path: Path to save model and training curves
        **optimizer_kwargs: Additional optimizer parameters
    """
    
    history = TrainingHistory()
    current_lr = learning_rate
    
    if verbose:
        print("🚀 Starting Bhaskar's Advanced CNN Training")
        print(f"📊 Training samples: {len(x_train)}")
        if x_val is not None:
            print(f"📊 Validation samples: {len(x_val)}")
        print(f"⚙️  Optimizer: {optimizer.upper()}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎯 Initial learning rate: {learning_rate}")
        print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Training loop
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            batch_loss = train_batch(network, batch_x, batch_y, loss, loss_prime, 
                                   current_lr, optimizer, **optimizer_kwargs)
            epoch_loss += batch_loss
            num_batches += 1
        
        epoch_loss /= num_batches
        
        # Validation
        val_loss, val_acc = None, None
        if x_val is not None and y_val is not None:
            val_loss, val_acc = evaluate(network, x_val, y_val, loss)
        
        # Update learning rate
        if scheduler:
            current_lr = scheduler(epoch)
        
        # Update history
        history.update(epoch, epoch_loss, val_loss, None, val_acc, current_lr)
        
        # Print progress
        if verbose:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | " if val_loss else f"Loss: {epoch_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if early_stopping and val_loss is not None:
            if early_stopping(val_loss, network):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"✅ Training completed in {total_time:.2f} seconds")
        print(f"📈 Final train loss: {epoch_loss:.6f}")
        if val_loss is not None:
            print(f"📈 Final validation loss: {val_loss:.6f}")
            print(f"🎯 Final validation accuracy: {val_acc:.4f}")
    
    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save training curves
        history.plot_training_curves(os.path.join(save_path, 'training_curves.png'))
        
        # Save training history
        history_data = {
            'train_losses': history.train_losses,
            'val_losses': history.val_losses,
            'train_accuracies': history.train_accuracies,
            'val_accuracies': history.val_accuracies,
            'learning_rates': history.learning_rates,
            'epochs': history.epochs
        }
        
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"💾 Training results saved to {save_path}")
    
    return history

def save_model(network, filepath):
    """Save model weights to file"""
    model_data = {}
    for i, layer in enumerate(network):
        layer_data = {}
        if hasattr(layer, 'weights'):
            layer_data['weights'] = layer.weights.tolist()
        if hasattr(layer, 'bias'):
            layer_data['bias'] = layer.bias.tolist()
        if hasattr(layer, 'kernels'):
            layer_data['kernels'] = layer.kernels.tolist()
        if hasattr(layer, 'biases'):
            layer_data['biases'] = layer.biases.tolist()
        if hasattr(layer, 'gamma'):
            layer_data['gamma'] = layer.gamma.tolist()
        if hasattr(layer, 'beta'):
            layer_data['beta'] = layer.beta.tolist()
        
        model_data[f'layer_{i}'] = layer_data
    
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"💾 Model saved to {filepath}")

def load_model(network, filepath):
    """Load model weights from file"""
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    for i, layer in enumerate(network):
        layer_key = f'layer_{i}'
        if layer_key in model_data:
            layer_data = model_data[layer_key]
            
            if 'weights' in layer_data:
                layer.weights = np.array(layer_data['weights'])
            if 'bias' in layer_data:
                layer.bias = np.array(layer_data['bias'])
            if 'kernels' in layer_data:
                layer.kernels = np.array(layer_data['kernels'])
            if 'biases' in layer_data:
                layer.biases = np.array(layer_data['biases'])
            if 'gamma' in layer_data:
                layer.gamma = np.array(layer_data['gamma'])
            if 'beta' in layer_data:
                layer.beta = np.array(layer_data['beta'])
    
    print(f"📂 Model loaded from {filepath}")