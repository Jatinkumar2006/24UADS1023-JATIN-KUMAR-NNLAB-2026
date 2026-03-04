# Assignment 5
# CNN on Fashion MNIST using Keras
# Demonstrating effect of:
# Filter Size, Regularization, Batch Size, Optimizer

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers

# LOAD DATASET

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# MODEL BUILDER

def build_model(filter_size=3, regularization=None, optimizer="adam"):

    model = models.Sequential()

    # Input Layer
    model.add(layers.Input(shape=(28, 28, 1)))

    # Regularization selection
    if regularization == "l2":
        reg = regularizers.l2(0.001)
    else:
        reg = None

    # Convolution Layer
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(filter_size, filter_size),
        activation="relu",
        kernel_regularizer=reg
    ))

    model.add(layers.MaxPooling2D((2, 2)))

    if regularization == "dropout":
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# TRAIN & EVALUATE

def train_and_evaluate(filter_size=3,
                       regularization=None,
                       batch_size=64,
                       optimizer="adam",
                       epochs=5):

    print("\n===================================================")
    print(f"Filter={filter_size} | Reg={regularization} | "
          f"Batch={batch_size} | Optimizer={optimizer}")
    print("===================================================")

    model = build_model(filter_size, regularization, optimizer)

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Confusion Matrix
    predictions = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, predictions)

    return history, test_acc, cm


# PLOT FUNCTION (No Cut Titles)

def plot_results(history, cm, title):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Confusion Matrix
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="coolwarm",
                ax=axes[2],
                cbar=False)
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.subplots_adjust(top=0.85)
    plt.show()


# 1: FILTER SIZE

print("\n===== EXPERIMENT: FILTER SIZE =====")
filter_sizes = [3, 5, 7]

for fs in filter_sizes:
    history, acc, cm = train_and_evaluate(filter_size=fs)
    plot_results(history, cm, f"Effect of Filter Size ({fs}x{fs})")


# 2: REGULARIZATION

print("\n===== EXPERIMENT: REGULARIZATION =====")
regularizations = [None, "l2", "dropout"]

for reg in regularizations:
    history, acc, cm = train_and_evaluate(regularization=reg)
    plot_results(history, cm, f"Effect of Regularization ({reg})")


# 3: BATCH SIZE

print("\n===== EXPERIMENT: BATCH SIZE =====")
batch_sizes = [32, 64, 128]

for bs in batch_sizes:
    history, acc, cm = train_and_evaluate(batch_size=bs)
    plot_results(history, cm, f"Effect of Batch Size ({bs})")


# 4: OPTIMIZER

print("\n===== EXPERIMENT: OPTIMIZER =====")
optimizers = ["adam", "sgd", "rmsprop"]

for opt in optimizers:
    history, acc, cm = train_and_evaluate(optimizer=opt)
    plot_results(history, cm, f"Effect of Optimizer ({opt})")
