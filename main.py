"""
Bhaskar's Advanced CNN Implementation
====================================
A sophisticated Convolutional Neural Network built from scratch with:
- Modern CNN architecture with multiple layers
- Advanced activation functions (ReLU, Swish, GELU)
- Batch normalization and dropout for regularization
- Multiple loss functions and optimizers
- Comprehensive training framework with visualization
- Model saving/loading capabilities

This implementation demonstrates professional-grade deep learning concepts
and serves as a comprehensive learning resource.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
import os

# Import our advanced modules
from dense import Dense
from convolutional import Convolutional, MaxPooling, AveragePooling
from reshape import Reshape
from activation import ReLU, LeakyReLU, Swish, GELU, Mish, Sigmoid
from advanced_layers import BatchNormalization, Dropout
from losses import (categorical_cross_entropy, categorical_cross_entropy_prime,
                   focal_loss, focal_loss_prime, mean_squared_error, mean_squared_error_prime)
from network import (train, predict, save_model, load_model, 
                    EarlyStopping, LearningRateScheduler)

def preprocess_mnist_data(x, y, num_classes=10, limit=None):
    """Preprocess MNIST data with advanced normalization"""
    if limit:
        # Limit samples for faster training
        indices = np.random.choice(len(x), limit, replace=False)
        x, y = x[indices], y[indices]
    
    # Normalize and reshape
    x = x.reshape(len(x), 1, 28, 28).astype("float32") / 255.0
    
    # Apply advanced normalization
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    # One-hot encode labels
    y = np_utils.to_categorical(y, num_classes)
    y = y.reshape(len(y), num_classes, 1)
    
    return x, y

def preprocess_cifar10_data(x, y, limit=None):
    """Preprocess CIFAR-10 data with advanced normalization"""
    if limit:
        indices = np.random.choice(len(x), limit, replace=False)
        x, y = x[indices], y[indices]
    
    # Normalize and reshape (CIFAR-10 is already 32x32x3)
    x = x.astype("float32") / 255.0
    
    # Apply advanced normalization
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    # One-hot encode labels
    y = np_utils.to_categorical(y, 10)
    y = y.reshape(len(y), 10, 1)
    
    return x, y

def create_advanced_cnn_architecture(input_shape, num_classes, architecture='deep'):
    """Create advanced CNN architectures"""
    
    if architecture == 'deep':
        # Deep CNN with modern features
        network = [
            # First Convolutional Block
            Convolutional(input_shape, kernel_size=3, depth=32, padding='same', initialization='he'),
            BatchNormalization(input_shape),
            ReLU(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Convolutional((32, 14, 14), kernel_size=3, depth=64, padding='same', initialization='he'),
            BatchNormalization((32, 14, 14)),
            Swish(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Convolutional((64, 7, 7), kernel_size=3, depth=128, padding='same', initialization='he'),
            BatchNormalization((64, 7, 7)),
            GELU(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.25),
            
            # Flatten
            Reshape((128, 3, 3), (128 * 3 * 3, 1)),
            
            # Dense Layers
            Dense(128 * 3 * 3, 512, initialization='he', l2_reg=0.001),
            BatchNormalization((512, 1)),
            Mish(),
            Dropout(0.5),
            
            Dense(512, 256, initialization='he', l2_reg=0.001),
            BatchNormalization((256, 1)),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),
            
            Dense(256, num_classes, initialization='xavier'),
            Sigmoid()
        ]
    
    elif architecture == 'lightweight':
        # Lightweight CNN for faster training
        network = [
            Convolutional(input_shape, kernel_size=5, depth=16, padding='same', initialization='he'),
            ReLU(),
            MaxPooling(pool_size=2, stride=2),
            
            Convolutional((16, 14, 14), kernel_size=5, depth=32, padding='same', initialization='he'),
            ReLU(),
            MaxPooling(pool_size=2, stride=2),
            
            Reshape((32, 7, 7), (32 * 7 * 7, 1)),
            
            Dense(32 * 7 * 7, 128, initialization='he'),
            ReLU(),
            Dropout(0.5),
            
            Dense(128, num_classes, initialization='xavier'),
            Sigmoid()
        ]
    
    elif architecture == 'modern':
        # Modern CNN with attention-like features
        network = [
            Convolutional(input_shape, kernel_size=3, depth=64, padding='same', initialization='he'),
            BatchNormalization(input_shape),
            Swish(),
            Convolutional((64, 28, 28), kernel_size=3, depth=64, padding='same', initialization='he'),
            BatchNormalization((64, 28, 28)),
            Swish(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.2),
            
            Convolutional((64, 14, 14), kernel_size=3, depth=128, padding='same', initialization='he'),
            BatchNormalization((64, 14, 14)),
            GELU(),
            Convolutional((128, 14, 14), kernel_size=3, depth=128, padding='same', initialization='he'),
            BatchNormalization((128, 14, 14)),
            GELU(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.3),
            
            Convolutional((128, 7, 7), kernel_size=3, depth=256, padding='same', initialization='he'),
            BatchNormalization((128, 7, 7)),
            Mish(),
            Convolutional((256, 7, 7), kernel_size=3, depth=256, padding='same', initialization='he'),
            BatchNormalization((256, 7, 7)),
            Mish(),
            MaxPooling(pool_size=2, stride=2),
            Dropout(0.4),
            
            Reshape((256, 3, 3), (256 * 3 * 3, 1)),
            
            Dense(256 * 3 * 3, 512, initialization='he', l2_reg=0.001),
            BatchNormalization((512, 1)),
            Swish(),
            Dropout(0.5),
            
            Dense(512, num_classes, initialization='xavier'),
            Sigmoid()
        ]
    
    return network

def visualize_predictions(network, x_test, y_test, num_samples=10):
    """Visualize model predictions"""
    plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # Get prediction
        output = predict(network, x_test[i])
        prediction = np.argmax(output)
        true_label = np.argmax(y_test[i])
        
        # Plot image
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {prediction}, True: {true_label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training pipeline"""
    print("🚀 Bhaskar's Advanced CNN from Scratch")
    print("=" * 50)
    
    # Configuration
    DATASET = 'mnist'  # 'mnist' or 'cifar10'
    ARCHITECTURE = 'deep'  # 'deep', 'lightweight', 'modern'
    LIMIT_SAMPLES = 1000  # Limit for faster training
    
    # Load and preprocess data
    print(f"📊 Loading {DATASET.upper()} dataset...")
    
    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, y_train = preprocess_mnist_data(x_train, y_train, limit=LIMIT_SAMPLES)
        x_test, y_test = preprocess_mnist_data(x_test, y_test, limit=LIMIT_SAMPLES//5)
        input_shape = (1, 28, 28)
        num_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, y_train = preprocess_cifar10_data(x_train, y_train.ravel(), limit=LIMIT_SAMPLES)
        x_test, y_test = preprocess_cifar10_data(x_test, y_test.ravel(), limit=LIMIT_SAMPLES//5)
        input_shape = (3, 32, 32)
        num_classes = 10
    
    # Create train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Training samples: {len(x_train)}")
    print(f"   Validation samples: {len(x_val)}")
    print(f"   Test samples: {len(x_test)}")
    
    # Create network architecture
    print(f"\n🏗️  Building {ARCHITECTURE} CNN architecture...")
    network = create_advanced_cnn_architecture(input_shape, num_classes, ARCHITECTURE)
    print(f"✅ Network created with {len(network)} layers")
    
    # Training configuration
    config = {
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',  # 'sgd', 'adam', 'rmsprop'
        'scheduler': LearningRateScheduler.cosine_annealing(0.001, 50),
        'early_stopping': EarlyStopping(patience=10, min_delta=0.001),
        'save_path': f'results/{DATASET}_{ARCHITECTURE}_cnn'
    }
    
    print(f"\n⚙️  Training Configuration:")
    for key, value in config.items():
        if key != 'save_path':
            print(f"   {key}: {value}")
    
    # Train the model
    print(f"\n🎯 Starting training...")
    history = train(
        network=network,
        loss=categorical_cross_entropy,
        loss_prime=categorical_cross_entropy_prime,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        **config
    )
    
    # Test the model
    print(f"\n🧪 Testing model performance...")
    test_loss, test_acc = evaluate(network, x_test, y_test, categorical_cross_entropy)
    print(f"📈 Test Loss: {test_loss:.6f}")
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    
    # Visualize predictions
    print(f"\n👁️  Visualizing predictions...")
    visualize_predictions(network, x_test, y_test)
    
    # Save the trained model
    print(f"\n💾 Saving trained model...")
    save_model(network, os.path.join(config['save_path'], 'model_weights.json'))
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Results saved to: {config['save_path']}")

def demo_different_architectures():
    """Demo function to showcase different architectures"""
    print("🎭 Bhaskar's CNN Architecture Showcase")
    print("=" * 50)
    
    # Load a small sample of MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train, limit=500)
    x_test, y_test = preprocess_mnist_data(x_test, y_test, limit=100)
    
    architectures = ['lightweight', 'deep', 'modern']
    results = {}
    
    for arch in architectures:
        print(f"\n🏗️  Testing {arch} architecture...")
        
        # Create network
        network = create_advanced_cnn_architecture((1, 28, 28), 10, arch)
        
        # Quick training (few epochs)
        history = train(
            network=network,
            loss=categorical_cross_entropy,
            loss_prime=categorical_cross_entropy_prime,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            epochs=5,
            learning_rate=0.01,
            batch_size=16,
            verbose=False
        )
        
        # Test performance
        test_loss, test_acc = evaluate(network, x_test, y_test, categorical_cross_entropy)
        results[arch] = {'loss': test_loss, 'accuracy': test_acc}
        
        print(f"   Final Test Accuracy: {test_acc:.4f}")
    
    # Compare results
    print(f"\n📊 Architecture Comparison:")
    print("-" * 40)
    for arch, metrics in results.items():
        print(f"{arch:12} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    # Run main training
    main()
    
    # Uncomment to run architecture demo
    # demo_different_architectures()
