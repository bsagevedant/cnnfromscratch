"""
Bhaskar's CNN Demo Scripts
==========================
Demonstration scripts showcasing different aspects of the CNN implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
import time

# Import our modules
from main import (preprocess_mnist_data, create_advanced_cnn_architecture, 
                 visualize_predictions)
from network import train, predict, EarlyStopping, LearningRateScheduler, evaluate
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from utils import (visualize_data_distribution, analyze_model_predictions,
                  plot_confusion_matrix, compare_architectures,
                  calculate_model_complexity, save_experiment_results)

def demo_activation_functions():
    """Demonstrate different activation functions"""
    print("🎭 Activation Functions Demo")
    print("=" * 40)
    
    from activation import ReLU, LeakyReLU, Swish, GELU, Mish, Sigmoid, Tanh
    
    # Create sample data
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'ReLU': ReLU(),
        'Leaky ReLU': LeakyReLU(0.01),
        'Swish': Swish(),
        'GELU': GELU(),
        'Mish': Mish(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh()
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, activation) in enumerate(activations.items()):
        # Forward pass
        y = activation.forward(x.reshape(-1, 1)).flatten()
        
        # Derivative
        grad = activation.backward(np.ones_like(x.reshape(-1, 1)), 1.0).flatten()
        
        # Plot activation
        plt.subplot(3, 3, i + 1)
        plt.plot(x, y, 'b-', linewidth=2, label=name)
        plt.plot(x, grad, 'r--', linewidth=2, label=f'{name} Derivative')
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Activation functions demo completed!")

def demo_loss_functions():
    """Demonstrate different loss functions"""
    print("🎯 Loss Functions Demo")
    print("=" * 40)
    
    from losses import (binary_cross_entropy, categorical_cross_entropy, 
                       mean_squared_error, huber_loss, focal_loss)
    
    # Create sample predictions and targets
    np.random.seed(42)
    
    # Binary classification
    y_true_binary = np.array([[1], [0], [1], [0], [1]])
    y_pred_binary = np.array([[0.9], [0.1], [0.8], [0.2], [0.7]])
    
    # Multi-class classification
    y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y_pred_cat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], 
                          [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    
    # Regression
    y_true_reg = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred_reg = np.array([[1.1], [1.9], [3.2], [3.8], [5.1]])
    
    # Calculate losses
    losses = {
        'Binary Cross-Entropy': binary_cross_entropy(y_true_binary, y_pred_binary),
        'Categorical Cross-Entropy': categorical_cross_entropy(y_true_cat, y_pred_cat),
        'Mean Squared Error': mean_squared_error(y_true_reg, y_pred_reg),
        'Huber Loss (δ=1.0)': huber_loss(y_true_reg, y_pred_reg, delta=1.0),
        'Focal Loss (α=1.0, γ=2.0)': focal_loss(y_true_binary, y_pred_binary, alpha=1.0, gamma=2.0)
    }
    
    # Visualize losses
    plt.figure(figsize=(12, 6))
    names = list(losses.keys())
    values = list(losses.values())
    
    bars = plt.bar(names, values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    plt.title('Loss Function Comparison')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Loss functions demo completed!")

def demo_architecture_comparison():
    """Compare different CNN architectures"""
    print("🏗️  Architecture Comparison Demo")
    print("=" * 40)
    
    # Load small dataset for quick comparison
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train, limit=500)
    x_test, y_test = preprocess_mnist_data(x_test, y_test, limit=100)
    
    # Split training data
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    architectures = ['lightweight', 'deep', 'modern']
    results = {}
    
    for arch in architectures:
        print(f"\n🔧 Training {arch} architecture...")
        start_time = time.time()
        
        # Create network
        network = create_advanced_cnn_architecture((1, 28, 28), 10, arch)
        
        # Calculate model complexity
        complexity = calculate_model_complexity(network)
        
        # Train network
        history = train(
            network=network,
            loss=categorical_cross_entropy,
            loss_prime=categorical_cross_entropy_prime,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=10,  # Quick training for demo
            learning_rate=0.01,
            batch_size=16,
            verbose=False
        )
        
        # Test performance
        test_loss, test_acc = evaluate(network, x_test, y_test, categorical_cross_entropy)
        
        training_time = time.time() - start_time
        
        results[arch] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'parameters': complexity['total_params'],
            'training_time': training_time,
            'history': history
        }
        
        print(f"   ✅ {arch}: {test_acc:.4f} accuracy, {training_time:.1f}s")
    
    # Visualize comparison
    compare_architectures(results, 'accuracy')
    
    # Save results
    save_experiment_results(results, 'results/architecture_comparison.json')
    
    print("\n✅ Architecture comparison demo completed!")
    return results

def demo_training_techniques():
    """Demonstrate advanced training techniques"""
    print("🎓 Advanced Training Techniques Demo")
    print("=" * 40)
    
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train, limit=800)
    x_test, y_test = preprocess_mnist_data(x_test, y_test, limit=200)
    
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create network
    network = create_advanced_cnn_architecture((1, 28, 28), 10, 'deep')
    
    # Different training configurations
    configs = {
        'Basic SGD': {
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'scheduler': None,
            'early_stopping': None
        },
        'Adam Optimizer': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'scheduler': None,
            'early_stopping': None
        },
        'With Learning Rate Schedule': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'scheduler': LearningRateScheduler.cosine_annealing(0.001, 20),
            'early_stopping': None
        },
        'With Early Stopping': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'scheduler': LearningRateScheduler.cosine_annealing(0.001, 50),
            'early_stopping': EarlyStopping(patience=5, min_delta=0.001)
        }
    }
    
    histories = []
    labels = []
    
    for config_name, config in configs.items():
        print(f"\n🔧 Training with {config_name}...")
        
        # Reset network weights
        network = create_advanced_cnn_architecture((1, 28, 28), 10, 'deep')
        
        # Train
        history = train(
            network=network,
            loss=categorical_cross_entropy,
            loss_prime=categorical_cross_entropy_prime,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=20,
            batch_size=32,
            verbose=False,
            **config
        )
        
        histories.append(history)
        labels.append(config_name)
        
        # Test performance
        test_loss, test_acc = evaluate(network, x_test, y_test, categorical_cross_entropy)
        print(f"   📈 Test Accuracy: {test_acc:.4f}")
    
    # Compare training histories
    plot_training_comparison(histories, labels)
    
    print("\n✅ Training techniques demo completed!")

def demo_data_analysis():
    """Demonstrate data analysis and visualization"""
    print("📊 Data Analysis Demo")
    print("=" * 40)
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train, limit=1000)
    x_test, y_test = preprocess_mnist_data(x_test, y_test, limit=200)
    
    # Split data
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Visualize data distribution
    visualize_data_distribution(y_train, y_val, title="MNIST Data Distribution")
    
    # Train a model for analysis
    print("\n🔧 Training model for analysis...")
    network = create_advanced_cnn_architecture((1, 28, 28), 10, 'lightweight')
    
    history = train(
        network=network,
        loss=categorical_cross_entropy,
        loss_prime=categorical_cross_entropy_prime,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=15,
        learning_rate=0.01,
        batch_size=32,
        verbose=False
    )
    
    # Analyze predictions
    analysis = analyze_model_predictions(network, x_test, y_test, num_samples=50)
    
    # Plot confusion matrix
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(analysis['true_labels'], analysis['predictions'], 
                         class_names=class_names, title='MNIST Classification Results')
    
    # Visualize some predictions
    visualize_predictions(network, x_test, y_test, num_samples=10)
    
    print("\n✅ Data analysis demo completed!")

def run_all_demos():
    """Run all demonstration scripts"""
    print("🎬 Bhaskar's CNN Complete Demo Suite")
    print("=" * 50)
    
    demos = [
        ("Activation Functions", demo_activation_functions),
        ("Loss Functions", demo_loss_functions),
        ("Architecture Comparison", demo_architecture_comparison),
        ("Training Techniques", demo_training_techniques),
        ("Data Analysis", demo_data_analysis)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            demo_func()
        except Exception as e:
            print(f"❌ Error in {demo_name}: {str(e)}")
            continue
    
    print(f"\n🎉 All demos completed!")
    print("📁 Check the 'results' directory for saved outputs")

if __name__ == "__main__":
    # Run individual demos or all demos
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        
        if demo_name == 'activations':
            demo_activation_functions()
        elif demo_name == 'losses':
            demo_loss_functions()
        elif demo_name == 'architectures':
            demo_architecture_comparison()
        elif demo_name == 'training':
            demo_training_techniques()
        elif demo_name == 'analysis':
            demo_data_analysis()
        else:
            print("Available demos: activations, losses, architectures, training, analysis, all")
    else:
        # Run all demos
        run_all_demos()
