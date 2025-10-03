"""
dev's Advanced Dense Layer Implementation
============================================
Enhanced fully connected layer with modern optimization features:
- Multiple weight initialization strategies
- Momentum-based optimization
- Gradient clipping
- L1/L2 regularization
- Adaptive learning rates
"""

import numpy as np
from layer import Layer

class Dense(Layer):
    """Advanced Dense (Fully Connected) Layer with optimization features"""
    
    def __init__(self, input_size, output_size, initialization='he', use_bias=True,
                 l1_reg=0.0, l2_reg=0.0, gradient_clip=1.0):
        """
        Initialize dense layer with advanced options
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            initialization: Weight initialization method ('he', 'xavier', 'random')
            use_bias: Whether to use bias terms
            l1_reg: L1 regularization coefficient
            l2_reg: L2 regularization coefficient
            gradient_clip: Gradient clipping threshold
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.gradient_clip = gradient_clip
        
        # Initialize weights
        self.weights = self._initialize_weights(initialization)
        
        # Initialize biases
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))
        else:
            self.bias = None
        
        # Momentum for optimization
        self.weight_momentum = np.zeros_like(self.weights)
        if self.use_bias:
            self.bias_momentum = np.zeros_like(self.bias)
        
        # For adaptive learning rates
        self.weight_cache = np.zeros_like(self.weights)
        if self.use_bias:
            self.bias_cache = np.zeros_like(self.bias)
    
    def _initialize_weights(self, initialization):
        """Initialize weights using different strategies"""
        if initialization == 'he':
            # He initialization for ReLU networks
            std = np.sqrt(2.0 / self.input_size)
            return np.random.normal(0, std, (self.output_size, self.input_size))
        elif initialization == 'xavier':
            # Xavier/Glorot initialization
            std = np.sqrt(2.0 / (self.input_size + self.output_size))
            return np.random.normal(0, std, (self.output_size, self.input_size))
        elif initialization == 'lecun':
            # LeCun initialization
            std = np.sqrt(1.0 / self.input_size)
            return np.random.normal(0, std, (self.output_size, self.input_size))
        else:  # random
            return np.random.randn(self.output_size, self.input_size) * 0.1

    def forward(self, input):
        """Forward pass with optional bias"""
        self.input = input
        
        # Linear transformation
        output = np.dot(self.weights, self.input)
        
        # Add bias if enabled
        if self.use_bias:
            output += self.bias
        
        return output

    def backward(self, output_gradient, learning_rate, momentum=0.9, 
                 optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Backward pass with advanced optimization
        
        Args:
            output_gradient: Gradient from next layer
            learning_rate: Learning rate
            momentum: Momentum for SGD
            optimizer: Optimization algorithm ('sgd', 'adam', 'rmsprop')
            beta1, beta2: Adam hyperparameters
            epsilon: Small constant for numerical stability
        """
        # Compute gradients
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        if self.use_bias:
            bias_gradient = output_gradient
        else:
            bias_gradient = None
        
        # Add regularization gradients
        if self.l1_reg > 0:
            l1_grad = self.l1_reg * np.sign(self.weights)
            weights_gradient += l1_grad
        
        if self.l2_reg > 0:
            l2_grad = self.l2_reg * self.weights
            weights_gradient += l2_grad
        
        # Gradient clipping
        weights_gradient = np.clip(weights_gradient, -self.gradient_clip, self.gradient_clip)
        if bias_gradient is not None:
            bias_gradient = np.clip(bias_gradient, -self.gradient_clip, self.gradient_clip)
        
        # Update weights based on optimizer
        if optimizer == 'sgd':
            self._update_sgd(weights_gradient, bias_gradient, learning_rate, momentum)
        elif optimizer == 'adam':
            self._update_adam(weights_gradient, bias_gradient, learning_rate, beta1, beta2, epsilon)
        elif optimizer == 'rmsprop':
            self._update_rmsprop(weights_gradient, bias_gradient, learning_rate, beta1, epsilon)
        else:
            # Default SGD update
            self.weights -= learning_rate * weights_gradient
            if self.use_bias:
                self.bias -= learning_rate * bias_gradient
        
        # Compute input gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        return input_gradient
    
    def _update_sgd(self, weights_gradient, bias_gradient, learning_rate, momentum):
        """SGD with momentum update"""
        self.weight_momentum = momentum * self.weight_momentum + learning_rate * weights_gradient
        self.weights -= self.weight_momentum
        
        if self.use_bias:
            self.bias_momentum = momentum * self.bias_momentum + learning_rate * bias_gradient
            self.bias -= self.bias_momentum
    
    def _update_adam(self, weights_gradient, bias_gradient, learning_rate, beta1, beta2, epsilon):
        """Adam optimizer update"""
        # Update biased first moment estimate
        self.weight_momentum = beta1 * self.weight_momentum + (1 - beta1) * weights_gradient
        if self.use_bias:
            self.bias_momentum = beta1 * self.bias_momentum + (1 - beta1) * bias_gradient
        
        # Update biased second raw moment estimate
        self.weight_cache = beta2 * self.weight_cache + (1 - beta2) * (weights_gradient ** 2)
        if self.use_bias:
            self.bias_cache = beta2 * self.bias_cache + (1 - beta2) * (bias_gradient ** 2)
        
        # Bias correction
        weight_momentum_corrected = self.weight_momentum / (1 - beta1)
        weight_cache_corrected = self.weight_cache / (1 - beta2)
        
        # Update weights
        self.weights -= learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + epsilon)
        
        if self.use_bias:
            bias_momentum_corrected = self.bias_momentum / (1 - beta1)
            bias_cache_corrected = self.bias_cache / (1 - beta2)
            self.bias -= learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected) + epsilon)
    
    def _update_rmsprop(self, weights_gradient, bias_gradient, learning_rate, beta, epsilon):
        """RMSprop optimizer update"""
        # Update cache
        self.weight_cache = beta * self.weight_cache + (1 - beta) * (weights_gradient ** 2)
        if self.use_bias:
            self.bias_cache = beta * self.bias_cache + (1 - beta) * (bias_gradient ** 2)
        
        # Update weights
        self.weights -= learning_rate * weights_gradient / (np.sqrt(self.weight_cache) + epsilon)
        if self.use_bias:
            self.bias -= learning_rate * bias_gradient / (np.sqrt(self.bias_cache) + epsilon)
    
    def get_regularization_loss(self):
        """Compute regularization loss"""
        loss = 0.0
        
        if self.l1_reg > 0:
            loss += self.l1_reg * np.sum(np.abs(self.weights))
        
        if self.l2_reg > 0:
            loss += self.l2_reg * np.sum(self.weights ** 2)
        
        return loss

class Linear(Dense):
    """Alias for Dense layer - more common in PyTorch style"""
    pass

class FullyConnected(Dense):
    """Alias for Dense layer"""
    pass