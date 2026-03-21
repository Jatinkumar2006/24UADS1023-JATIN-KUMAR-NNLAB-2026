"""
📌 File: predict_driver.py

🎯 Purpose:
Test a CSV file using trained models (LSTM, GRU, Transformer)

👉 No UI required (offline mode)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 🔹 CONFIG: Give your CSV path here
# -------------------------------
# -------------------------------
# 🔹 Get CSV Path from User
# -------------------------------
import os

CSV_PATH = input("📂 Enter CSV file path: ").strip()

# Validate path
if not os.path.exists(CSV_PATH):
    print("❌ Error: File not found. Please check the path.")
    exit()

if not CSV_PATH.endswith(".csv"):
    print("❌ Error: Please provide a CSV file.")
    exit()

print(f"✅ File found: {CSV_PATH}\n")
# -------------------------------
# 🔹 Load Models
# -------------------------------
print("🔄 Loading models...")

lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
gru_model = tf.keras.models.load_model("models/gru_model.h5")
transformer_model = tf.keras.models.load_model("models/transformer_model.h5")

print("✅ Models loaded successfully.\n")

# -------------------------------
# 🔹 Feature Engineering
# -------------------------------
def create_features(df):
    df["acc_mag"] = np.sqrt(df["X_Acc"]**2 + df["Y_Acc"]**2 + df["Z_Acc"]**2)
    df["gyro_mag"] = np.sqrt(df["X_Gyro"]**2 + df["Y_Gyro"]**2 + df["Z_Gyro"]**2)
    df["jerk"] = df["acc_mag"].diff().abs().fillna(0)
    return df

# -------------------------------
# 🔹 Sequence Creation
# -------------------------------
def create_sequences(df, window_size=100, step_size=25):
    
    feature_cols = [
        'X_Acc','Y_Acc','Z_Acc',
        'X_Gyro','Y_Gyro','Z_Gyro',
        'acc_mag','gyro_mag','jerk'
    ]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X = []
    
    for i in range(0, len(df) - window_size, step_size):
        X.append(df[feature_cols].values[i:i+window_size])
    
    return np.array(X)

# -------------------------------
# 🔹 MAIN FUNCTION
# -------------------------------
def predict_driver(csv_path):
    
    print(f"📂 Loading file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Keep only required columns
    df = df[['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']]
    
    # Feature Engineering
    df = create_features(df)
    
    # Create sequences
    X = create_sequences(df)
    
    if len(X) == 0:
        print("❌ Not enough data (minimum 50 rows required)")
        return
    
    print(f"🔢 Total sequences created: {len(X)}\n")
    
    # -------------------------------
    # 🔹 Predictions
    # -------------------------------
    models = {
        "LSTM": lstm_model,
        "GRU": gru_model,
        "Transformer": transformer_model
    }
    
    results = {}
    
    print("🤖 MODEL RESULTS")
    print("="*40)
    
    for name, model in models.items():
        preds = model.predict(X, verbose=0)
        preds = np.argmax(preds, axis=1) + 1
        
        avg_rating = np.mean(preds)
        results[name] = avg_rating
        
        print(f"{name} Average Rating: ⭐ {avg_rating:.2f}")
    
    # -------------------------------
    # 🔹 Final Score
    # -------------------------------
    final_score = np.mean(list(results.values()))
    
    print("\n" + "="*40)
    print(f"🏆 FINAL DRIVING SCORE: ⭐ {final_score:.2f} / 5")
    print("="*40)
    
    # -------------------------------
    # 🔹 Interpretation (5 LEVEL)
    # -------------------------------
    
    if final_score >= 4.5:
        print("🌟 Excellent Driver (5⭐)")
        print("🚀 Very Safe – You can confidently ride with this driver.")
    
    elif final_score >= 3.5:
        print("👍 Good Driver (4⭐)")
        print("✅ Safe driving behavior observed.")
    
    elif final_score >= 2.5:
        print("😐 Normal Driver (3⭐)")
        print("⚖️ Moderate driving – sometimes safe, sometimes aggressive.")
    
    elif final_score >= 1.5:
        print("⚠️ Rash Driver (2⭐)")
        print("❗ Risky driving detected – be cautious.")
    
    else:
        print("❌ Dangerous Driver (1⭐)")
        print("🚫 Highly unsafe – not recommended to ride.")

# -------------------------------
# 🔹 RUN
# -------------------------------
if __name__ == "__main__":
    predict_driver(CSV_PATH)