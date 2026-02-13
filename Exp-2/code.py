import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

plt.ion()

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sigmoid Function Graph (Theory Visualization)
x_vals = np.linspace(-10, 10, 200)
y_vals = sigmoid(x_vals)

plt.figure()
plt.plot(x_vals, y_vals)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

# XOR Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],[1],[1],[0]])

# MLP Architecture
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.5
epochs = 5000

loss_history = []
accuracy_history = []

# Training Loop
for epoch in range(epochs):

    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)

    # Loss (Mean Squared Error)
    loss = np.mean((y - output)**2)
    loss_history.append(loss)

    # Accuracy
    predictions = (output > 0.5).astype(int)
    acc = accuracy_score(y, predictions)
    accuracy_history.append(acc)

    # Backpropagation
    d_output = (output - y) * sigmoid_derivative(output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Update weights
    W2 -= hidden_output.T.dot(d_output) * learning_rate
    b2 -= np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 -= X.T.dot(d_hidden) * learning_rate
    b1 -= np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print every 500 epochs
    if epoch % 500 == 0:
        print(f"\nEpoch {epoch}")
        print("Input | Target | Predicted | Loss")
        print("-" * 40)
        for i in range(len(X)):
            print(f"{X[i]} |   {y[i][0]}    |   {predictions[i][0]}    | {loss:.4f}")

plt.ioff()

# Final Evaluation
hidden_output = sigmoid(np.dot(X, W1) + b1)
output = sigmoid(np.dot(hidden_output, W2) + b2)
predictions = (output > 0.5).astype(int)

print("\nFinal Predictions (Probabilities):")
print(output)

print("\nRounded Predictions:")
print(predictions)

print("\nFinal Accuracy:", accuracy_score(y, predictions)*100, "%")
print("Confusion Matrix:\n", confusion_matrix(y, predictions))

# Loss vs Epoch
plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Accuracy vs Epoch
plt.figure()
plt.plot(accuracy_history)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# Decision Boundary Plot
plt.figure()

xx, yy = np.meshgrid(np.linspace(-0.5,1.5,100),
                     np.linspace(-0.5,1.5,100))

grid = np.c_[xx.ravel(), yy.ravel()]

hidden = sigmoid(np.dot(grid, W1) + b1)
final = sigmoid(np.dot(hidden, W2) + b2)

Z = final.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.3)

for i in range(len(X)):
    if y[i] == 1:
        plt.scatter(X[i][0], X[i][1], s=100)
    else:
        plt.scatter(X[i][0], X[i][1], s=100)

plt.title("MLP Decision Boundary (XOR)")
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.grid()
plt.show()
