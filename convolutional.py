"""
dev's Advanced Convolutional Neural Network Layer
===================================================
A sophisticated implementation of convolutional layers with modern features:
- Multiple padding strategies (valid, same, full)
- Stride support for efficient computation
- He/Xavier initialization
- Batch processing optimization
- Gradient clipping for stability
"""

import numpy as np
from layer import Layer
from scipy import signal

class Convolutional(Layer):
    """Advanced Convolutional Layer with modern features"""
    
    def __init__(self, input_shape, kernel_size, depth, stride=1, padding='valid', 
                 initialization='he', use_bias=True):
        """
        Initialize convolutional layer with advanced options
        
        Args:
            input_shape: (depth, height, width) of input
            kernel_size: Size of convolutional kernel (assumes square)
            depth: Number of output feature maps
            stride: Step size for convolution
            padding: 'valid', 'same', or 'full'
            initialization: 'he', 'xavier', or 'random'
            use_bias: Whether to use bias terms
        """
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        # Calculate output dimensions based on padding
        if padding == 'valid':
            output_height = (input_height - kernel_size) // stride + 1
            output_width = (input_width - kernel_size) // stride + 1
            self.pad_height, self.pad_width = 0, 0
        elif padding == 'same':
            output_height = input_height // stride
            output_width = input_width // stride
            self.pad_height = max(0, (kernel_size - 1) // 2)
            self.pad_width = max(0, (kernel_size - 1) // 2)
        elif padding == 'full':
            output_height = (input_height + kernel_size - 1) // stride
            output_width = (input_width + kernel_size - 1) // stride
            self.pad_height = kernel_size - 1
            self.pad_width = kernel_size - 1
        else:
            raise ValueError(f"Unknown padding type: {padding}")
            
        self.output_shape = (depth, output_height, output_width)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        # Initialize weights using specified method
        self.kernels = self._initialize_weights(initialization)
        
        # Initialize biases
        if self.use_bias:
            self.biases = np.zeros(self.output_shape)
        else:
            self.biases = None
            
        # For momentum-based optimization
        self.kernel_momentum = np.zeros_like(self.kernels)
        if self.use_bias:
            self.bias_momentum = np.zeros_like(self.biases)
    
    def _initialize_weights(self, initialization):
        """Initialize weights using different strategies"""
        if initialization == 'he':
            # He initialization for ReLU networks
            fan_in = self.input_depth * self.kernel_size * self.kernel_size
            std = np.sqrt(2.0 / fan_in)
            return np.random.normal(0, std, self.kernels_shape)
        elif initialization == 'xavier':
            # Xavier/Glorot initialization
            fan_in = self.input_depth * self.kernel_size * self.kernel_size
            fan_out = self.depth * self.kernel_size * self.kernel_size
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0, std, self.kernels_shape)
        else:  # random
            return np.random.randn(*self.kernels_shape) * 0.1
    
    def _pad_input(self, input):
        """Add padding to input if needed"""
        if self.pad_height == 0 and self.pad_width == 0:
            return input
        
        padded = np.zeros((
            self.input_depth,
            self.input_shape[1] + 2 * self.pad_height,
            self.input_shape[2] + 2 * self.pad_width
        ))
        padded[:, self.pad_height:self.pad_height+self.input_shape[1],
               self.pad_width:self.pad_width+self.input_shape[2]] = input
        return padded

    def forward(self, input):
        """Forward pass with optimized convolution"""
        self.input = input
        
        # Add padding if needed
        if self.padding != 'valid':
            padded_input = self._pad_input(input)
        else:
            padded_input = input
        
        # Initialize output
        if self.use_bias:
            self.output = np.copy(self.biases)
        else:
            self.output = np.zeros(self.output_shape)
        
        # Perform convolution
        for i in range(self.depth):
            for j in range(self.input_depth):
                if self.stride == 1:
                    # Use scipy for stride=1 (faster)
                    self.output[i] += signal.correlate2d(
                        padded_input[j], self.kernels[i, j], "valid"
                    )
                else:
                    # Manual implementation for stride > 1
                    for y in range(0, padded_input.shape[1] - self.kernel_size + 1, self.stride):
                        for x in range(0, padded_input.shape[2] - self.kernel_size + 1, self.stride):
                            self.output[i, y//self.stride, x//self.stride] += np.sum(
                                padded_input[j, y:y+self.kernel_size, x:x+self.kernel_size] * 
                                self.kernels[i, j]
                            )
        
        return self.output

    def backward(self, output_gradient, learning_rate, momentum=0.9, gradient_clip=1.0):
        """Backward pass with momentum and gradient clipping"""
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        
        # Add padding to input gradient if needed
        if self.padding != 'valid':
            padded_input = self._pad_input(self.input)
            padded_gradient = np.zeros_like(padded_input)
        else:
            padded_input = self.input
            padded_gradient = input_gradient
        
        # Compute gradients
        for i in range(self.depth):
            for j in range(self.input_depth):
                if self.stride == 1:
                    # Kernel gradient
                    kernels_gradient[i, j] = signal.correlate2d(
                        padded_input[j], output_gradient[i], "valid"
                    )
                    # Input gradient
                    padded_gradient[j] += signal.convolve2d(
                        output_gradient[i], self.kernels[i, j], "full"
                    )
                else:
                    # Manual implementation for stride > 1
                    for y in range(0, padded_input.shape[1] - self.kernel_size + 1, self.stride):
                        for x in range(0, padded_input.shape[2] - self.kernel_size + 1, self.stride):
                            # Kernel gradient
                            kernels_gradient[i, j] += output_gradient[i, y//self.stride, x//self.stride] * \
                                                    padded_input[j, y:y+self.kernel_size, x:x+self.kernel_size]
                            # Input gradient
                            padded_gradient[j, y:y+self.kernel_size, x:x+self.kernel_size] += \
                                output_gradient[i, y//self.stride, x//self.stride] * self.kernels[i, j]
        
        # Gradient clipping
        kernels_gradient = np.clip(kernels_gradient, -gradient_clip, gradient_clip)
        
        # Update weights with momentum
        self.kernel_momentum = momentum * self.kernel_momentum + learning_rate * kernels_gradient
        self.kernels -= self.kernel_momentum
        
        # Update biases with momentum
        if self.use_bias:
            bias_gradient = output_gradient
            bias_gradient = np.clip(bias_gradient, -gradient_clip, gradient_clip)
            self.bias_momentum = momentum * self.bias_momentum + learning_rate * bias_gradient
            self.biases -= self.bias_momentum
        
        # Remove padding from input gradient if it was added
        if self.padding != 'valid':
            input_gradient = padded_gradient[:, self.pad_height:self.pad_height+self.input_shape[1],
                                           self.pad_width:self.pad_width+self.input_shape[2]]
        
        return input_gradient

class MaxPooling(Layer):
    """Max Pooling Layer for CNNs"""
    
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, input):
        """Forward pass for max pooling"""
        self.input = input
        batch_size, channels, height, width = input.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(input)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for y in range(out_height):
                    for x in range(out_width):
                        start_y = y * self.stride
                        start_x = x * self.stride
                        end_y = start_y + self.pool_size
                        end_x = start_x + self.pool_size
                        
                        pool_region = input[b, c, start_y:end_y, start_x:end_x]
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        
                        self.output[b, c, y, x] = pool_region[max_idx]
                        self.mask[b, c, start_y + max_idx[0], start_x + max_idx[1]] = 1
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for max pooling"""
        input_gradient = np.zeros_like(self.input)
        
        # Propagate gradients only to the positions that were selected during forward pass
        input_gradient[self.mask == 1] = output_gradient.flatten()
        
        return input_gradient

class AveragePooling(Layer):
    """Average Pooling Layer for CNNs"""
    
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, input):
        """Forward pass for average pooling"""
        self.input = input
        batch_size, channels, height, width = input.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for y in range(out_height):
                    for x in range(out_width):
                        start_y = y * self.stride
                        start_x = x * self.stride
                        end_y = start_y + self.pool_size
                        end_x = start_x + self.pool_size
                        
                        pool_region = input[b, c, start_y:end_y, start_x:end_x]
                        self.output[b, c, y, x] = np.mean(pool_region)
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass for average pooling"""
        input_gradient = np.zeros_like(self.input)
        batch_size, channels, height, width = self.input.shape
        
        # Distribute gradients equally across all positions in each pool
        for b in range(batch_size):
            for c in range(channels):
                for y in range(output_gradient.shape[2]):
                    for x in range(output_gradient.shape[3]):
                        start_y = y * self.stride
                        start_x = x * self.stride
                        end_y = start_y + self.pool_size
                        end_x = start_x + self.pool_size
                        
                        grad = output_gradient[b, c, y, x] / (self.pool_size * self.pool_size)
                        input_gradient[b, c, start_y:end_y, start_x:end_x] += grad
        
        return input_gradient