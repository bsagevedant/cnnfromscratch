"""
Bhaskar's Advanced Loss Functions Module
=======================================
Comprehensive collection of loss functions for deep learning:
- Classification losses (Cross-entropy variants, Focal Loss)
- Regression losses (MSE, MAE, Huber)
- Advanced losses (Label Smoothing, Dice Loss)
- Numerical stability improvements
"""

import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy with numerical stability"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy derivative with numerical stability"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Categorical cross-entropy with numerical stability"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_cross_entropy_prime(y_true, y_pred, epsilon=1e-15):
    """Categorical cross-entropy derivative"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / np.size(y_true)

def sparse_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Sparse categorical cross-entropy"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.log(y_pred[range(len(y_true)), y_true]))

def sparse_categorical_cross_entropy_prime(y_true, y_pred, epsilon=1e-15):
    """Sparse categorical cross-entropy derivative"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    grad = np.zeros_like(y_pred)
    grad[range(len(y_true)), y_true] = -1.0 / y_pred[range(len(y_true)), y_true]
    return grad / np.size(y_true)

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_prime(y_true, y_pred):
    """MSE derivative"""
    return 2 * (y_pred - y_true) / np.size(y_true)

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error loss"""
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_error_prime(y_true, y_pred):
    """MAE derivative"""
    return np.sign(y_pred - y_true) / np.size(y_true)

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss - robust to outliers"""
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def huber_loss_prime(y_true, y_pred, delta=1.0):
    """Huber loss derivative"""
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    
    squared_grad = error
    linear_grad = delta * np.sign(error)
    
    return np.where(is_small_error, squared_grad, linear_grad) / np.size(y_true)

def focal_loss(y_true, y_pred, alpha=1.0, gamma=2.0, epsilon=1e-15):
    """Focal Loss for handling class imbalance"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate focal weight
    focal_weight = alpha * (1 - y_pred) ** gamma
    
    # Calculate cross-entropy
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    return np.mean(focal_weight * ce)

def focal_loss_prime(y_true, y_pred, alpha=1.0, gamma=2.0, epsilon=1e-15):
    """Focal Loss derivative"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate focal weight and its derivative
    focal_weight = alpha * (1 - y_pred) ** gamma
    focal_weight_prime = -alpha * gamma * (1 - y_pred) ** (gamma - 1)
    
    # Calculate cross-entropy and its derivative
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    ce_prime = (1 - y_true) / (1 - y_pred) - y_true / y_pred
    
    # Focal loss derivative
    return (focal_weight_prime * ce + focal_weight * ce_prime) / np.size(y_true)

def dice_loss(y_true, y_pred, smooth=1e-15):
    """Dice Loss for segmentation tasks"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice

def dice_loss_prime(y_true, y_pred, smooth=1e-15):
    """Dice Loss derivative"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    numerator = 2 * y_true * (union + smooth)
    denominator = (union + smooth) ** 2
    
    return -(numerator - 2 * intersection * y_pred) / denominator

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-15):
    """Tversky Loss - generalization of Dice Loss"""
    intersection = np.sum(y_true * y_pred)
    false_positive = np.sum((1 - y_true) * y_pred)
    false_negative = np.sum(y_true * (1 - y_pred))
    
    tversky = (intersection + smooth) / (intersection + alpha * false_positive + beta * false_negative + smooth)
    return 1 - tversky

def tversky_loss_prime(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-15):
    """Tversky Loss derivative"""
    intersection = np.sum(y_true * y_pred)
    false_positive = np.sum((1 - y_true) * y_pred)
    false_negative = np.sum(y_true * (1 - y_pred))
    
    denominator = (intersection + alpha * false_positive + beta * false_negative + smooth) ** 2
    
    numerator = y_true * (intersection + alpha * false_positive + beta * false_negative + smooth) - \
                (intersection + smooth) * (y_true - beta * y_true + alpha * (1 - y_true))
    
    return -numerator / denominator

def label_smoothing_cross_entropy(y_true, y_pred, smoothing=0.1, epsilon=1e-15):
    """Label Smoothing Cross-Entropy"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Apply label smoothing
    num_classes = y_pred.shape[-1]
    smoothed_labels = (1 - smoothing) * y_true + smoothing / num_classes
    
    return -np.mean(np.sum(smoothed_labels * np.log(y_pred), axis=-1))

def label_smoothing_cross_entropy_prime(y_true, y_pred, smoothing=0.1, epsilon=1e-15):
    """Label Smoothing Cross-Entropy derivative"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Apply label smoothing
    num_classes = y_pred.shape[-1]
    smoothed_labels = (1 - smoothing) * y_true + smoothing / num_classes
    
    return -smoothed_labels / y_pred / np.size(y_true)

def kullback_leibler_divergence(y_true, y_pred, epsilon=1e-15):
    """KL Divergence loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=-1))

def kullback_leibler_divergence_prime(y_true, y_pred, epsilon=1e-15):
    """KL Divergence derivative"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / np.size(y_true)

def cosine_similarity_loss(y_true, y_pred):
    """Cosine Similarity Loss"""
    # Normalize vectors
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=-1, keepdims=True) + 1e-8)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    cosine_sim = np.sum(y_true_norm * y_pred_norm, axis=-1)
    
    return np.mean(1 - cosine_sim)

def cosine_similarity_loss_prime(y_true, y_pred):
    """Cosine Similarity Loss derivative"""
    # Normalize vectors
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=-1, keepdims=True) + 1e-8)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    cosine_sim = np.sum(y_true_norm * y_pred_norm, axis=-1, keepdims=True)
    
    # Gradient
    y_pred_norm_grad = (y_true_norm - cosine_sim * y_pred_norm) / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-8)
    
    return -y_pred_norm_grad / np.size(y_true)

# Loss function dictionary for easy access
LOSS_FUNCTIONS = {
    'binary_crossentropy': (binary_cross_entropy, binary_cross_entropy_prime),
    'categorical_crossentropy': (categorical_cross_entropy, categorical_cross_entropy_prime),
    'sparse_categorical_crossentropy': (sparse_categorical_cross_entropy, sparse_categorical_cross_entropy_prime),
    'mse': (mean_squared_error, mean_squared_error_prime),
    'mae': (mean_absolute_error, mean_absolute_error_prime),
    'huber': (huber_loss, huber_loss_prime),
    'focal': (focal_loss, focal_loss_prime),
    'dice': (dice_loss, dice_loss_prime),
    'tversky': (tversky_loss, tversky_loss_prime),
    'label_smoothing': (label_smoothing_cross_entropy, label_smoothing_cross_entropy_prime),
    'kl_divergence': (kullback_leibler_divergence, kullback_leibler_divergence_prime),
    'cosine_similarity': (cosine_similarity_loss, cosine_similarity_loss_prime)
}