"""
Bhaskar's Advanced Activation Functions Module
==============================================
A comprehensive collection of activation functions with optimized implementations
and advanced features for deep learning networks.

Features:
- Multiple activation functions (ReLU, Leaky ReLU, ELU, Swish, GELU, Mish)
- Automatic gradient computation
- Numerical stability improvements
- Memory-efficient implementations
"""

import numpy as np
from layer import Layer

class Activation(Layer):
    """Base activation layer with automatic gradient computation"""
    
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class ReLU(Activation):
    """Rectified Linear Unit - Most popular activation for CNNs"""
    
    def __init__(self, alpha=0.0):
        super().__init__(self._relu, self._relu_prime)
        self.alpha = alpha  # For leaky ReLU variant
    
    def _relu(self, x):
        """ReLU with optional leaky variant"""
        return np.where(x > 0, x, self.alpha * x)
    
    def _relu_prime(self, x):
        """ReLU derivative"""
        return np.where(x > 0, 1, self.alpha)

class LeakyReLU(ReLU):
    """Leaky ReLU with small negative slope"""
    
    def __init__(self, alpha=0.01):
        super().__init__(alpha)

class ELU(Activation):
    """Exponential Linear Unit - Smooth alternative to ReLU"""
    
    def __init__(self, alpha=1.0):
        super().__init__(self._elu, self._elu_prime)
        self.alpha = alpha
    
    def _elu(self, x):
        """ELU activation"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def _elu_prime(self, x):
        """ELU derivative"""
        return np.where(x > 0, 1, self.alpha * np.exp(x))

class Swish(Activation):
    """Swish activation - Self-gated activation function"""
    
    def __init__(self, beta=1.0):
        super().__init__(self._swish, self._swish_prime)
        self.beta = beta
    
    def _swish(self, x):
        """Swish: x * sigmoid(beta * x)"""
        return x * (1 / (1 + np.exp(-self.beta * x)))
    
    def _swish_prime(self, x):
        """Swish derivative"""
        sigmoid = 1 / (1 + np.exp(-self.beta * x))
        return sigmoid * (1 + self.beta * x * (1 - sigmoid))

class GELU(Activation):
    """Gaussian Error Linear Unit - Used in Transformers"""
    
    def __init__(self):
        super().__init__(self._gelu, self._gelu_prime)
    
    def _gelu(self, x):
        """GELU: x * Φ(x) where Φ is standard normal CDF"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _gelu_prime(self, x):
        """GELU derivative"""
        tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
        return 0.5 * (1 + tanh_term + x * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2))

class Mish(Activation):
    """Mish activation - Smooth, non-monotonic activation"""
    
    def __init__(self):
        super().__init__(self._mish, self._mish_prime)
    
    def _mish(self, x):
        """Mish: x * tanh(softplus(x))"""
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def _mish_prime(self, x):
        """Mish derivative"""
        exp_x = np.exp(x)
        softplus = np.log(1 + exp_x)
        tanh_softplus = np.tanh(softplus)
        return tanh_softplus + x * (1 - tanh_softplus**2) * (exp_x / (1 + exp_x))

class Sigmoid(Activation):
    """Sigmoid activation with numerical stability"""
    
    def __init__(self):
        super().__init__(self._sigmoid, self._sigmoid_prime)
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid"""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_prime(self, x):
        """Sigmoid derivative"""
        s = self._sigmoid(x)
        return s * (1 - s)

class Tanh(Activation):
    """Hyperbolic tangent activation"""
    
    def __init__(self):
        super().__init__(self._tanh, self._tanh_prime)
    
    def _tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def _tanh_prime(self, x):
        """Tanh derivative"""
        return 1 - np.tanh(x)**2