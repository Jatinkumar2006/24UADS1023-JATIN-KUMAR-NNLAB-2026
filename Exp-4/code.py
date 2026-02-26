import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

tf.random.set_seed(42)

# LOAD DATASET

(ds_train_raw, ds_test_raw), _ = tfds.load(
    "mnist",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])
    label = tf.one_hot(label, depth=10)
    return image, label

# HE INITIALIZATION

def he_init(shape):
    return tf.Variable(
        tf.random.normal(shape, stddev=tf.sqrt(2.0 / shape[0]))
    )

# BUILD NETWORK
def build_network(hidden_size):

    W1 = he_init([784, hidden_size])
    b1 = tf.Variable(tf.zeros([hidden_size]))

    W2 = he_init([hidden_size, hidden_size // 2])
    b2 = tf.Variable(tf.zeros([hidden_size // 2]))

    W3 = he_init([hidden_size // 2, 10])
    b3 = tf.Variable(tf.zeros([10]))

    return [W1, b1, W2, b2, W3, b3]


# FORWARD PASS

def forward_pass(X, params, activation):

    W1, b1, W2, b2, W3, b3 = params

    act_fn = {
        "relu": tf.nn.relu,
        "sigmoid": tf.nn.sigmoid,
        "tanh": tf.nn.tanh
    }[activation]

    A1 = act_fn(tf.matmul(X, W1) + b1)
    A2 = act_fn(tf.matmul(A1, W2) + b2)

    return tf.matmul(A2, W3) + b3


# TRAINING FUNCTION

def train_and_evaluate(activation="relu",
                       hidden_size=256,
                       lr=0.01,
                       batch_size=128,
                       epochs=10):

    params = build_network(hidden_size)

    ds_train = ds_train_raw.map(preprocess).shuffle(60000).batch(batch_size)
    ds_test  = ds_test_raw.map(preprocess).batch(batch_size)

    train_acc_history = []
    train_loss_history = []

    print(f"\nConfiguration → Activation={activation}, "
          f"Hidden={hidden_size}, LR={lr}, "
          f"Batch={batch_size}, Epochs={epochs}")

    for epoch in range(epochs):

        correct = 0
        total = 0
        epoch_loss = 0.0
        steps = 0

        for X_batch, y_batch in ds_train:

            with tf.GradientTape() as tape:
                logits = forward_pass(X_batch, params, activation)

                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_batch,
                        logits=logits
                    )
                )

            grads = tape.gradient(loss, params)

            for p, g in zip(params, grads):
                if g is not None:
                    p.assign_sub(lr * g)

            epoch_loss += loss.numpy()
            steps += 1

            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            labels = tf.argmax(y_batch, axis=1, output_type=tf.int32)

            correct += tf.reduce_sum(
                tf.cast(preds == labels, tf.float32)
            ).numpy()

            total += X_batch.shape[0]

        avg_loss = epoch_loss / steps
        accuracy = correct / total

        train_loss_history.append(avg_loss)
        train_acc_history.append(accuracy)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    # ================= TEST EVALUATION =================

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in ds_test:
        logits = forward_pass(X_batch, params, activation)

        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        labels = tf.argmax(y_batch, axis=1, output_type=tf.int32)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

        correct += tf.reduce_sum(
            tf.cast(preds == labels, tf.float32)
        ).numpy()

        total += X_batch.shape[0]

    test_acc = correct / total

    print(f"  --> Test Accuracy: {test_acc:.4f}")

    return test_acc, train_acc_history, train_loss_history, all_preds, all_labels


def plot_experiment(title, param_name, results):

    n = len(results)

    # =========================
    # ACCURACY CURVES
    # =========================
    fig1, axes1 = plt.subplots(1, n, figsize=(5*n, 4))

    if n == 1:
        axes1 = [axes1]

    for ax, (label, test_acc, acc_hist, _, _, _) in zip(axes1, results):
        ax.plot(acc_hist, linewidth=2)
        ax.set_title(f"{param_name} = {label}\nTest Acc: {test_acc:.4f}",
                     fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.7, 1.0)
        ax.grid(alpha=0.3)

    fig1.suptitle(f"Effect of {title} — Accuracy Curves",
                  fontsize=14,
                  fontweight="bold")

    fig1.tight_layout(rect=[0, 0, 1, 0.93])   # 🔥 FIX TITLE CUTTING
    plt.show()


    # =========================
    # CONFUSION MATRICES
    # =========================
    fig2, axes2 = plt.subplots(1, n, figsize=(5*n, 4))

    if n == 1:
        axes2 = [axes2]

    for ax, (label, test_acc, _, _, preds, labels) in zip(axes2, results):
        cm = confusion_matrix(labels, preds)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10),
            cbar=False,
            annot_kws={"size": 7},
            ax=ax
        )

        ax.set_title(f"{param_name} = {label}\nTest Acc: {test_acc:.4f}",
                     fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig2.suptitle(f"Effect of {title} — Confusion Matrices",
                  fontsize=14,
                  fontweight="bold")

    fig2.tight_layout(rect=[0, 0, 1, 0.93])   # 🔥 FIX TITLE CUTTING
    plt.show()
# EXPERIMENTS

# A) Activation
print("\n===== Varying Activation Function =====")
activations = ["relu", "sigmoid", "tanh"]
act_data = []
act_results = {}

for act in activations:
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(
        activation=act
    )
    act_results[act] = test_acc
    act_data.append((act, test_acc, acc_hist, loss_hist, preds, labels))

plot_experiment("Activation Function", "Activation", act_data)


# B) Hidden Size
print("\n===== Varying Hidden Layer Size =====")
hidden_sizes = [64, 256, 512]
size_data = []
size_results = {}

for hs in hidden_sizes:
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(
        hidden_size=hs
    )
    size_results[hs] = test_acc
    size_data.append((hs, test_acc, acc_hist, loss_hist, preds, labels))

plot_experiment("Hidden Layer Size", "Hidden", size_data)


# C) Learning Rate
print("\n===== Varying Learning Rate =====")
learning_rates = [0.001, 0.01, 0.1]
lr_data = []
lr_results = {}

for lr in learning_rates:
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(
        lr=lr
    )
    lr_results[lr] = test_acc
    lr_data.append((lr, test_acc, acc_hist, loss_hist, preds, labels))

plot_experiment("Learning Rate", "LR", lr_data)
