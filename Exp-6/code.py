# ==========================================
# Improved RNN Time Series Prediction
# (Matches friend's style output)
# ==========================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Generate Data (Sine Wave)
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)

# Normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# 2. Create Sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 30   # Increased for smoother prediction
X, y = create_sequences(data, seq_length)

X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# 3. Train / Validation / Test Split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# 4. RNN Model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64)
        
        out, _ = self.rnn(x, h0)
        
        out = out[:, -1, :]   # last time step
        out = self.fc(out)
        
        return out

model = RNNModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    
    # ---- Training ----
    model.train()
    train_output = model(X_train)
    train_loss = criterion(train_output, y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # ---- Validation ----
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

# 6. Plot Train vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, linestyle="dashed", label="Val Loss")
plt.title("Training & Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 7. Testing / Prediction
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Convert to numpy
predictions = predictions.numpy()
y_test_np = y_test.numpy()

# Inverse scaling
predictions = scaler.inverse_transform(predictions)
y_test_np = scaler.inverse_transform(y_test_np)

# 8. Smooth Predictions (OPTIONAL BUT IMPORTANT)
def moving_average(data, window_size=5):
    return np.convolve(data.flatten(), np.ones(window_size)/window_size, mode='same')

smooth_predictions = moving_average(predictions)

# 9. Plot Predictions vs Actual
plt.figure(figsize=(10,5))

plt.plot(y_test_np[:200], label="Actual")
plt.plot(smooth_predictions[:200], linestyle="dashed", label="Predicted")

plt.title("Predictions vs Actuals (first 200 test samples)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# 10. Final Loss
print("\nFinal Training Loss:", train_losses[-1])
print("Final Validation Loss:", val_losses[-1])