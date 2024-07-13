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
hidden_layer_size = 10

# Initialize Weights and Biases
hidden_layer_a_weights = np.random.rand(num_features, hidden_layer_size) - 0.5
hidden_layer_a_bias = np.random.rand(1, hidden_layer_size) - 0.5
hidden_layer_b_weights = np.random.rand(hidden_layer_size, hidden_layer_size) - 0.5
hidden_layer_b_bias = np.random.rand(1, hidden_layer_size) - 0.5
output_layer_weights = np.random.rand(hidden_layer_size, num_classes) - 0.5
output_layer_bias = np.random.rand(1, num_classes) - 0.5

# Training Loop
for i in range(iterations):
    # Forward Propagation
    hidden_layer_a_unactivated = training_data.dot(hidden_layer_a_weights) + hidden_layer_a_bias
    hidden_layer_a_activated = np.maximum(hidden_layer_a_unactivated, 0)  # ReLU
    hidden_layer_b_unactivated = hidden_layer_a_activated.dot(hidden_layer_b_weights) + hidden_layer_b_bias
    hidden_layer_b_activated = np.maximum(hidden_layer_b_unactivated, 0)  # ReLU
    output_layer_unactivated = hidden_layer_b_activated.dot(output_layer_weights) + output_layer_bias
    output_layer_activated = np.exp(output_layer_unactivated) / np.sum(np.exp(output_layer_unactivated), axis=1,
                                                                       keepdims=True)  # softmax

    # Back Propagation
    output_layer_error = output_layer_activated - training_labels_one_hot
    output_layer_weight_gradient = hidden_layer_b_activated.T.dot(output_layer_error) / num_examples
    output_layer_bias_gradient = np.sum(output_layer_error, axis=0, keepdims=True) / num_examples

    hidden_layer_b_error = output_layer_error.dot(output_layer_weights.T) * (
            hidden_layer_b_unactivated > 0)  # ReLU derivative
    hidden_layer_b_weight_gradient = hidden_layer_a_activated.T.dot(hidden_layer_b_error) / num_examples
    hidden_layer_b_bias_gradient = np.sum(hidden_layer_b_error, axis=0, keepdims=True) / num_examples

    hidden_layer_a_error = hidden_layer_b_error.dot(hidden_layer_b_weights.T) * (
            hidden_layer_a_unactivated > 0)  # ReLU derivative
    hidden_layer_a_weight_gradient = training_data.T.dot(hidden_layer_a_error) / num_examples
    hidden_layer_a_bias_gradient = np.sum(hidden_layer_a_error, axis=0, keepdims=True) / num_examples

    # Update Parameters
    hidden_layer_a_weights -= learning_rate * hidden_layer_a_weight_gradient
    hidden_layer_a_bias -= learning_rate * hidden_layer_a_bias_gradient
    hidden_layer_b_weights -= learning_rate * hidden_layer_b_weight_gradient
    hidden_layer_b_bias -= learning_rate * hidden_layer_b_bias_gradient
    output_layer_weights -= learning_rate * output_layer_weight_gradient
    output_layer_bias -= learning_rate * output_layer_bias_gradient

    if i % 10 == 0:
        predictions = np.argmax(output_layer_activated, axis=1)
        accuracy = np.sum(predictions == training_labels) / training_labels.size
        print(f"Iteration {i}/{iterations}, Accuracy: {accuracy * 100:.2f}%")

predictions = np.argmax(output_layer_activated, axis=1)
accuracy = np.sum(predictions == training_labels) / training_labels.size
end = time.time()
print(f"Final Accuracy: {accuracy * 100:.2f}% Elapsed: {end - start:.2f} seconds.")

"""
Iteration 0/500, Accuracy: 5.56%
Iteration 10/500, Accuracy: 17.89%
Iteration 20/500, Accuracy: 21.63%
Iteration 30/500, Accuracy: 24.30%
Iteration 40/500, Accuracy: 26.86%
Iteration 50/500, Accuracy: 29.89%
Iteration 60/500, Accuracy: 32.59%
Iteration 70/500, Accuracy: 35.19%
Iteration 80/500, Accuracy: 37.67%
Iteration 90/500, Accuracy: 39.95%
Iteration 100/500, Accuracy: 42.51%
Iteration 110/500, Accuracy: 45.08%
Iteration 120/500, Accuracy: 47.00%
Iteration 130/500, Accuracy: 49.53%
Iteration 140/500, Accuracy: 52.29%
Iteration 150/500, Accuracy: 54.85%
Iteration 160/500, Accuracy: 57.48%
Iteration 170/500, Accuracy: 60.24%
Iteration 180/500, Accuracy: 62.84%
Iteration 190/500, Accuracy: 65.07%
Iteration 200/500, Accuracy: 66.91%
Iteration 210/500, Accuracy: 68.28%
Iteration 220/500, Accuracy: 69.40%
Iteration 230/500, Accuracy: 70.38%
Iteration 240/500, Accuracy: 71.26%
Iteration 250/500, Accuracy: 71.91%
Iteration 260/500, Accuracy: 68.31%
Iteration 270/500, Accuracy: 72.01%
Iteration 280/500, Accuracy: 73.32%
Iteration 290/500, Accuracy: 73.68%
Iteration 300/500, Accuracy: 73.69%
Iteration 310/500, Accuracy: 74.22%
Iteration 320/500, Accuracy: 74.94%
Iteration 330/500, Accuracy: 75.49%
Iteration 340/500, Accuracy: 75.88%
Iteration 350/500, Accuracy: 76.28%
Iteration 360/500, Accuracy: 76.68%
Iteration 370/500, Accuracy: 77.14%
Iteration 380/500, Accuracy: 77.53%
Iteration 390/500, Accuracy: 77.90%
Iteration 400/500, Accuracy: 78.29%
Iteration 410/500, Accuracy: 78.71%
Iteration 420/500, Accuracy: 79.13%
Iteration 430/500, Accuracy: 79.51%
Iteration 440/500, Accuracy: 79.83%
Iteration 450/500, Accuracy: 79.93%
Iteration 460/500, Accuracy: 79.68%
Iteration 470/500, Accuracy: 80.14%
Iteration 480/500, Accuracy: 80.77%
Iteration 490/500, Accuracy: 81.18%
Final Accuracy: 81.44% Elapsed: 36.54 seconds.

Process finished with exit code 0
"""