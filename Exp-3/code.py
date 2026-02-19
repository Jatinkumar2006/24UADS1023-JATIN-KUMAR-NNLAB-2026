import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# LOAD DATASET
def get_data():
    dataset,_ = tfds.load('mnist', with_info=True, as_supervised=True, split=['train','test'])

    def preprocess(img,label):
        img = tf.cast(img,tf.float32)/255.0
        img = tf.reshape(img,[784])
        return img, tf.cast(label,tf.int32)

    train = dataset[0].map(preprocess).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    test  = dataset[1].map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    return train,test

train_loader,test_loader = get_data()

# NETWORK PARAMETERS
initializer = tf.random_normal_initializer(stddev=0.1)

W1=tf.Variable(initializer([784,256]),dtype=tf.float32)
b1=tf.Variable(tf.zeros([256]),dtype=tf.float32)

W2=tf.Variable(initializer([256,128]),dtype=tf.float32)
b2=tf.Variable(tf.zeros([128]),dtype=tf.float32)

W3=tf.Variable(initializer([128,10]),dtype=tf.float32)
b3=tf.Variable(tf.zeros([10]),dtype=tf.float32)

params=[W1,b1,W2,b2,W3,b3]

# FEED FORWARD
def forward(x):
    h1=tf.nn.relu(tf.matmul(x,W1)+b1)
    h2=tf.nn.relu(tf.matmul(h1,W2)+b2)
    out=tf.matmul(h2,W3)+b3
    return out

# TRAINING SETTINGS
learning_rate=0.1
epochs=10

loss_history=[]
acc_history=[]

print("\nEpoch | Loss | Accuracy")
print("--------------------------")

# TRAINING LOOP
for epoch in range(epochs):

    avg_loss=tf.metrics.Mean()

    for images,labels in train_loader:

        with tf.GradientTape() as tape:
            logits=forward(images)

            loss=tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits
                )
            )

        grads=tape.gradient(loss,params)

        # manual backprop weight update
        for p,g in zip(params,grads):
            if isinstance(g,tf.IndexedSlices):
                g=tf.convert_to_tensor(g)
            p.assign_sub(learning_rate*g)

        avg_loss.update_state(loss)

    # EVALUATION
    preds_all=[]
    labels_all=[]

    for x,y in test_loader:
        logits=forward(x)
        preds=tf.argmax(logits,axis=1)
        preds_all.extend(preds.numpy())
        labels_all.extend(y.numpy())

    acc=np.mean(np.array(preds_all)==np.array(labels_all))

    epoch_loss=float(avg_loss.result().numpy())

    loss_history.append(epoch_loss)
    acc_history.append(acc)

    print(f"{epoch+1:>3}   | {epoch_loss:.4f} | {acc:.4f}")

# FINAL RESULTS
print("\nFinal Accuracy:",acc_history[-1])
print("Final Loss:",loss_history[-1])

# GRAPH 1 — LOSS
plt.figure()
plt.plot(loss_history,marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# GRAPH 2 — ACCURACY
plt.figure()
plt.plot(acc_history,marker='o')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# GRAPH 3 — CONFUSION MATRIX
cm=confusion_matrix(labels_all,preds_all)

plt.figure()
sns.heatmap(cm,annot=True,fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SAMPLE PREDICTIONS
plt.figure(figsize=(10,4))

sample_images=None
sample_preds=None
sample_labels=None

for x,y in test_loader:
    sample_images=x
    sample_labels=y
    sample_preds=tf.argmax(forward(x),axis=1)
    break

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(sample_images[i].numpy().reshape(28,28),cmap="gray")
    plt.title(f"P:{sample_preds[i].numpy()} T:{sample_labels[i].numpy()}")
    plt.axis("off")

plt.tight_layout()
plt.show()
