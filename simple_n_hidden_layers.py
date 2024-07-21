import time

import numpy as np
from mnist import MNIST

start = time.time()
# Hyperparameters
iterations = 500
learning_rate = 0.1

# Load MNIST Data
dataset = MNIST("MNIST_ORG")
training_data, training_labels = dataset.load_training()

# Preprocess Data
training_data = np.array(training_data) / 255.0
training_labels = np.array(training_labels)
training_labels_one_hot = np.zeros((training_labels.size, training_labels.max() + 1))
training_labels_one_hot[np.arange(training_labels.size), training_labels] = 1

# Network Architecture Parameters
num_examples, num_features = training_data.shape
num_classes = len(set(training_labels))
hidden_layer_size = 100
num_layers = 3

# Initialize Weights and Biases
layers = list()
# weights, bias, unactivated, activated, error, weight_gradient, bias_gradient
layers.append((None, None, None, training_data, None, None, None))

for i in range(num_layers):
    input_size = hidden_layer_size if i > 0 else num_features
    weights = np.random.rand(input_size, hidden_layer_size) - 0.5
    bias = np.random.rand(1, hidden_layer_size) - 0.5
    layers.append((weights, bias, None, None, None, None, None))

# Output layer
input_size = hidden_layer_size
weights = np.random.rand(input_size, num_classes) - 0.5
bias = np.random.rand(1, num_classes) - 0.5
layers.append((weights, bias, None, None, None, None, None))

# Training Loop
for i in range(iterations):
    # Forward Propagation
    for layer_idx in range(len(layers)):
        if layer_idx > 0:
            weights, bias, unactivated, activated, error, weight_gradient, bias_gradient = layers[layer_idx]
            previous_activated = layers[layer_idx - 1][3]
            unactivated = previous_activated.dot(weights) + bias
            activated = np.maximum(unactivated, 0) if layer_idx < len(layers) - 1 else np.exp(unactivated) / np.sum(
                np.exp(unactivated), axis=1, keepdims=True)
            layers[layer_idx] = (weights, bias, unactivated, activated, error, weight_gradient, bias_gradient)

    # Back Propagation
    for layer_idx in reversed(range(len(layers))):
        if layer_idx > 0:
            weights, bias, unactivated, activated, error, weight_gradient, bias_gradient = layers[layer_idx]
            if layer_idx == len(layers) - 1:
                error = activated - training_labels_one_hot
            else:
                next_layer_weights, _, _, _, next_layer_error, _, _ = layers[layer_idx + 1]
                error = next_layer_error.dot(next_layer_weights.T) * (unactivated > 0)
            previous_activated = layers[layer_idx - 1][3]
            weight_gradient = previous_activated.T.dot(error) / num_examples
            bias_gradient = np.sum(error, axis=0, keepdims=True) / num_examples
            layers[layer_idx] = (weights, bias, unactivated, activated, error, weight_gradient, bias_gradient)

    # Update Parameters
    for layer_idx in range(len(layers)):
        if layer_idx > 0:
            weights, bias, unactivated, activated, error, weight_gradient, bias_gradient = layers[layer_idx]
            weights = weights - learning_rate * weight_gradient
            bias = bias - learning_rate * bias_gradient
            layers[layer_idx] = (weights, bias, unactivated, activated, error, weight_gradient, bias_gradient)

    if i % 10 == 0:
        predictions = np.argmax(layers[-1][3], axis=1)
        accuracy = np.sum(predictions == training_labels) / training_labels.size
        print(f"Iteration {i}/{iterations}, Accuracy: {accuracy * 100:.2f}%")

predictions = np.argmax(layers[-1][3], axis=1)
accuracy = np.sum(predictions == training_labels) / training_labels.size
end = time.time()
print(f"Final Accuracy: {accuracy * 100:.2f}% Elapsed: {end - start:.2f} seconds.")

"""
Iteration 0/500, Accuracy: 8.39%
Iteration 10/500, Accuracy: 44.34%
Iteration 20/500, Accuracy: 57.51%
Iteration 30/500, Accuracy: 63.98%
Iteration 40/500, Accuracy: 67.91%
Iteration 50/500, Accuracy: 70.78%
Iteration 60/500, Accuracy: 73.10%
Iteration 70/500, Accuracy: 74.96%
Iteration 80/500, Accuracy: 76.42%
Iteration 90/500, Accuracy: 77.72%
Iteration 100/500, Accuracy: 78.76%
Iteration 110/500, Accuracy: 79.70%
Iteration 120/500, Accuracy: 80.52%
Iteration 130/500, Accuracy: 81.22%
Iteration 140/500, Accuracy: 81.94%
Iteration 150/500, Accuracy: 82.48%
Iteration 160/500, Accuracy: 83.05%
Iteration 170/500, Accuracy: 83.48%
Iteration 180/500, Accuracy: 83.95%
Iteration 190/500, Accuracy: 84.41%
Iteration 200/500, Accuracy: 84.76%
Iteration 210/500, Accuracy: 85.12%
Iteration 220/500, Accuracy: 85.47%
Iteration 230/500, Accuracy: 85.76%
Iteration 240/500, Accuracy: 86.03%
Iteration 250/500, Accuracy: 86.30%
Iteration 260/500, Accuracy: 86.56%
Iteration 270/500, Accuracy: 86.81%
Iteration 280/500, Accuracy: 87.04%
Iteration 290/500, Accuracy: 87.27%
Iteration 300/500, Accuracy: 87.48%
Iteration 310/500, Accuracy: 87.69%
Iteration 320/500, Accuracy: 87.87%
Iteration 330/500, Accuracy: 88.05%
Iteration 340/500, Accuracy: 88.25%
Iteration 350/500, Accuracy: 88.42%
Iteration 360/500, Accuracy: 88.59%
Iteration 370/500, Accuracy: 88.72%
Iteration 380/500, Accuracy: 88.87%
Iteration 390/500, Accuracy: 89.02%
Iteration 400/500, Accuracy: 89.17%
Iteration 410/500, Accuracy: 89.30%
Iteration 420/500, Accuracy: 89.45%
Iteration 430/500, Accuracy: 89.56%
Iteration 440/500, Accuracy: 89.69%
Iteration 450/500, Accuracy: 89.81%
Iteration 460/500, Accuracy: 89.91%
Iteration 470/500, Accuracy: 90.03%
Iteration 480/500, Accuracy: 90.12%
Iteration 490/500, Accuracy: 90.23%
Final Accuracy: 90.32% Elapsed: 162.52 seconds.
"""