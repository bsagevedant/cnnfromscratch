"""
dev's Advanced Neural Network Layers
========================================
Advanced layers for deep learning including:
- Batch Normalization
- Dropout
- Layer Normalization
- Group Normalization
- Attention mechanisms
"""

import numpy as np
from layer import Layer

class BatchNormalization(Layer):
    """Batch Normalization layer for stable training"""
    
    def __init__(self, input_shape, momentum=0.9, epsilon=1e-8):
        """
        Initialize batch normalization layer
        
        Args:
            input_shape: Shape of input tensor
            momentum: Momentum for running statistics
            epsilon: Small constant for numerical stability
        """
        self.input_shape = input_shape
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(input_shape)  # Scale parameter
        self.beta = np.zeros(input_shape)  # Shift parameter
        
        # Running statistics (updated during training)
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.ones(input_shape)
        
        # For backward pass
        self.normalized = None
        self.std = None
        
        self.training = True
    
    def forward(self, input):
        """Forward pass with batch normalization"""
        self.input = input
        
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Use batch statistics
            mean, var = batch_mean, batch_var
        else:
            # Use running statistics during inference
            mean, var = self.running_mean, self.running_var
        
        # Normalize
        self.std = np.sqrt(var + self.epsilon)
        self.normalized = (input - mean) / self.std
        
        # Scale and shift
        output = self.gamma * self.normalized + self.beta
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for batch normalization"""
        batch_size = self.input.shape[0]
        
        # Gradients for gamma and beta
        gamma_grad = np.sum(output_gradient * self.normalized, axis=0)
        beta_grad = np.sum(output_gradient, axis=0)
        
        # Gradient for normalized input
        normalized_grad = output_gradient * self.gamma
        
        # Gradient for variance
        var_grad = np.sum(normalized_grad * (self.input - np.mean(self.input, axis=0)) * 
                         -0.5 * (self.std ** -3), axis=0)
        
        # Gradient for mean
        mean_grad = np.sum(normalized_grad * (-1 / self.std), axis=0) + \
                   var_grad * np.mean(-2 * (self.input - np.mean(self.input, axis=0)), axis=0)
        
        # Gradient for input
        input_gradient = normalized_grad / self.std + \
                        var_grad * 2 * (self.input - np.mean(self.input, axis=0)) / batch_size + \
                        mean_grad / batch_size
        
        # Update parameters
        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad
        
        return input_gradient
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training

class Dropout(Layer):
    """Dropout layer for regularization"""
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize dropout layer
        
        Args:
            dropout_rate: Probability of setting a neuron to zero
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, input):
        """Forward pass with dropout"""
        if self.training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, input.shape) / (1 - self.dropout_rate)
            return input * self.mask
        else:
            # No dropout during inference
            return input
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for dropout"""
        if self.training:
            return output_gradient * self.mask
        else:
            return output_gradient
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training

class LayerNormalization(Layer):
    """Layer Normalization for stable training"""
    
    def __init__(self, input_shape, epsilon=1e-8):
        """
        Initialize layer normalization
        
        Args:
            input_shape: Shape of input tensor
            epsilon: Small constant for numerical stability
        """
        self.input_shape = input_shape
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
    
    def forward(self, input):
        """Forward pass with layer normalization"""
        self.input = input
        
        # Compute mean and variance across features
        mean = np.mean(input, axis=-1, keepdims=True)
        var = np.var(input, axis=-1, keepdims=True)
        
        # Normalize
        self.std = np.sqrt(var + self.epsilon)
        self.normalized = (input - mean) / self.std
        
        # Scale and shift
        output = self.gamma * self.normalized + self.beta
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for layer normalization"""
        # Gradients for gamma and beta
        gamma_grad = np.sum(output_gradient * self.normalized, axis=0)
        beta_grad = np.sum(output_gradient, axis=0)
        
        # Gradient for normalized input
        normalized_grad = output_gradient * self.gamma
        
        # Compute gradients for mean and variance
        var_grad = np.sum(normalized_grad * (self.input - np.mean(self.input, axis=-1, keepdims=True)) * 
                         -0.5 * (self.std ** -3), axis=-1, keepdims=True)
        
        mean_grad = np.sum(normalized_grad * (-1 / self.std), axis=-1, keepdims=True) + \
                   var_grad * np.mean(-2 * (self.input - np.mean(self.input, axis=-1, keepdims=True)), axis=-1, keepdims=True)
        
        # Gradient for input
        input_gradient = normalized_grad / self.std + \
                        var_grad * 2 * (self.input - np.mean(self.input, axis=-1, keepdims=True)) / self.input_shape[-1] + \
                        mean_grad / self.input_shape[-1]
        
        # Update parameters
        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad
        
        return input_gradient

class GroupNormalization(Layer):
    """Group Normalization for stable training"""
    
    def __init__(self, input_shape, num_groups=32, epsilon=1e-8):
        """
        Initialize group normalization
        
        Args:
            input_shape: Shape of input tensor
            num_groups: Number of groups for normalization
            epsilon: Small constant for numerical stability
        """
        self.input_shape = input_shape
        self.num_groups = num_groups
        self.epsilon = epsilon
        
        # Ensure channels are divisible by num_groups
        assert input_shape[0] % num_groups == 0, "Channels must be divisible by num_groups"
        
        self.channels_per_group = input_shape[0] // num_groups
        
        # Learnable parameters
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
    
    def forward(self, input):
        """Forward pass with group normalization"""
        self.input = input
        batch_size, channels, height, width = input.shape
        
        # Reshape for group processing
        input_reshaped = input.reshape(batch_size, self.num_groups, self.channels_per_group, height, width)
        
        # Compute mean and variance across groups
        mean = np.mean(input_reshaped, axis=(2, 3, 4), keepdims=True)
        var = np.var(input_reshaped, axis=(2, 3, 4), keepdims=True)
        
        # Normalize
        self.std = np.sqrt(var + self.epsilon)
        self.normalized = (input_reshaped - mean) / self.std
        
        # Reshape back
        self.normalized = self.normalized.reshape(batch_size, channels, height, width)
        
        # Scale and shift
        output = self.gamma * self.normalized + self.beta
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for group normalization"""
        # Gradients for gamma and beta
        gamma_grad = np.sum(output_gradient * self.normalized, axis=0)
        beta_grad = np.sum(output_gradient, axis=0)
        
        # Gradient for normalized input
        normalized_grad = output_gradient * self.gamma
        
        # Update parameters
        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad
        
        # Simplified gradient computation for input
        input_gradient = normalized_grad / self.std.reshape(self.input_shape)
        
        return input_gradient

class SelfAttention(Layer):
    """Self-Attention mechanism for sequence modeling"""
    
    def __init__(self, input_dim, num_heads=8, dropout_rate=0.1):
        """
        Initialize self-attention layer
        
        Args:
            input_dim: Input dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout_rate = dropout_rate
        
        assert input_dim % num_heads == 0, "Input dimension must be divisible by num_heads"
        
        # Weight matrices for Q, K, V
        self.W_q = np.random.randn(input_dim, input_dim) * 0.1
        self.W_k = np.random.randn(input_dim, input_dim) * 0.1
        self.W_v = np.random.randn(input_dim, input_dim) * 0.1
        self.W_o = np.random.randn(input_dim, input_dim) * 0.1
        
        # Dropout for attention weights
        self.dropout = Dropout(dropout_rate)
        
        self.training = True
    
    def forward(self, input):
        """Forward pass with self-attention"""
        self.input = input
        batch_size, seq_len, input_dim = input.shape
        
        # Compute Q, K, V
        Q = np.dot(input, self.W_q)
        K = np.dot(input, self.W_k)
        V = np.dot(input, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        self.attention_weights = self._softmax(scores)
        
        # Apply dropout to attention weights
        if self.training:
            self.attention_weights = self.dropout.forward(self.attention_weights)
        
        # Apply attention to values
        attended = np.matmul(self.attention_weights, V)
        
        # Reshape and concatenate heads
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, input_dim)
        
        # Output projection
        output = np.dot(attended, self.W_o)
        
        return output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for self-attention"""
        # Simplified backward pass - in practice, this would be more complex
        batch_size, seq_len, input_dim = output_gradient.shape
        
        # Gradient for output projection
        W_o_grad = np.dot(output_gradient.T, self.input)
        self.W_o -= learning_rate * W_o_grad
        
        # Gradient for input (simplified)
        input_gradient = np.dot(output_gradient, self.W_o.T)
        
        return input_gradient
    
    def _softmax(self, x):
        """Softmax function with numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
        self.dropout.set_training(training)
