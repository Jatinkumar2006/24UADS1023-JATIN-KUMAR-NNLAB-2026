import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

plt.ion()   # turn ON interactive plotting


# Activation function (Perceptron step function)
def step(value):
    return 1 if value >= 0 else 0


# Evaluate accuracy and confusion matrix
def evaluate_model(X, y, weights, gate_name):
    X_bias = np.c_[np.ones(X.shape[0]), X]

    predictions = []
    for i in range(len(X_bias)):
        predictions.append(step(np.dot(X_bias[i], weights)))

    accuracy = accuracy_score(y, predictions) * 100
    cm = confusion_matrix(y, predictions)

    print(f"\n Performance Evaluation ({gate_name})")
    print(f"Training Accuracy = {accuracy:.2f}%")
    print(f"Test Accuracy     = {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(cm)


# Plot input points once (green = 1, red = 0)
def plot_input_points(axis, X, y):
    for i in range(len(X)):
        color = "green" if y[i] == 1 else "red"
        axis.scatter(X[i][0], X[i][1], color=color, s=120)


#               NAND GATE

print("\n===== NAND GATE TRAINING =====")

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nand = np.array([1,1,1,0])

X_bias = np.c_[np.ones(X.shape[0]), X]

weights = np.zeros(X_bias.shape[1])
learning_rate = 0.1
epoch = 0

nand_error_history = []
nand_accuracy_history = []

while True:
    epoch += 1
    total_error = 0

    print(f"\nEpoch {epoch}")
    print("x1 x2 | target | pred | error | weights")
    print("-" * 55)

    # -------- Training on all samples --------
    for i in range(len(X_bias)):
        net_input = np.dot(X_bias[i], weights)
        prediction = step(net_input)

        error = y_nand[i] - prediction
        weights += learning_rate * error * X_bias[i]
        total_error += abs(error)

        print(f"{X[i][0]}  {X[i][1]}  |   {y_nand[i]}    |  {prediction}   |  {error}   | {weights}")

    # -------- Store metrics --------
    nand_error_history.append(total_error)

    preds = [step(np.dot(X_bias[i], weights)) for i in range(len(X_bias))]
    nand_accuracy_history.append(accuracy_score(y_nand, preds) * 100)

    # -------- Visualization --------
    plt.clf()
    ax = plt.gca()
    plot_input_points(ax, X, y_nand)

    if weights[2] != 0:
        x_vals = np.array([-0.5, 1.5])
        y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
        plt.plot(x_vals, y_vals)

    plt.title(f"NAND Gate | Epoch {epoch}")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid()
    plt.pause(1)

    if total_error == 0:
        print("\n NAND converged successfully!")
        break

plt.ioff()
plt.show()

evaluate_model(X, y_nand, weights, "NAND Gate")


# -------- NAND graphs --------
plt.figure()
plt.plot(nand_error_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.title("NAND Gate: Error vs Epoch")
plt.grid()
plt.show()

plt.figure()
plt.plot(nand_accuracy_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("NAND Gate: Accuracy vs Epoch")
plt.grid()
plt.show()


#               XOR GATE

print("\n===== XOR GATE TRAINING =====")
plt.ion()

class XORPerceptron:
    """
    A single-layer perceptron to demonstrate
    why XOR cannot be learned using a linear boundary
    """

    def __init__(self, learning_rate=0.1, max_updates=25):
        self.lr = learning_rate
        self.max_updates = max_updates
        self.final_weights = (0, 0, 0)
        self.error_history = []
        self.accuracy_history = []

    def train(self, X, y):
        w1, w2, bias = 0, 0, 0
        update_count = 0
        epoch = 1

        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid()
        plot_input_points(ax, X, y)

        x_vals = np.linspace(-0.5, 1.5, 100)
        line, = ax.plot(x_vals, np.zeros(100), linewidth=2)

        while update_count < self.max_updates:

            print(f"\nEpoch {epoch}")
            print("x1 x2 | target | pred | error | w1 w2 bias")
            print("-" * 55)

            epoch_error = 0

            for i in range(len(X)):
                x1, x2 = X[i]
                target = y[i]

                net_input = w1 * x1 + w2 * x2 + bias
                prediction = step(net_input)
                error = target - prediction
                epoch_error += abs(error)

                print(f"{x1}  {x2}  |   {target}    |  {prediction}   |  {error}   | "
                      f"{round(w1,2)} {round(w2,2)} {round(bias,2)}")

                if error != 0:
                    w1 += self.lr * error * x1
                    w2 += self.lr * error * x2
                    bias += self.lr * error
                    update_count += 1

                    y_vals = -(w1 * x_vals + bias) / (w2 + 1e-6)
                    line.set_data(x_vals, y_vals)
                    ax.set_title(f"XOR Gate | Epoch {epoch}")
                    plt.draw()
                    plt.pause(0.6)

                if update_count >= self.max_updates:
                    print("\n XOR is not linearly separable (training stopped)")
                    break

            self.error_history.append(epoch_error)

            preds = [step(w1 * X[i][0] + w2 * X[i][1] + bias) for i in range(len(X))]
            self.accuracy_history.append(accuracy_score(y, preds) * 100)

            epoch += 1

        self.final_weights = (w1, w2, bias)
        plt.ioff()
        plt.show()


# -------- Run XOR --------
y_xor = np.array([0, 1, 1, 0])
xor_model = XORPerceptron()
xor_model.train(X, y_xor)

if xor_model.final_weights is not None:
    w1, w2, b = xor_model.final_weights
    xor_weights = np.array([b, w1, w2])
else:
    xor_weights = np.array([0, 0, 0])

evaluate_model(X, y_xor, xor_weights, "XOR Gate")


# -------- XOR graphs --------
plt.figure()
plt.plot(xor_model.error_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.title("XOR Gate: Error vs Epoch")
plt.grid()
plt.show()

plt.figure()
plt.plot(xor_model.accuracy_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("XOR Gate: Accuracy vs Epoch")
plt.grid()
plt.show()
