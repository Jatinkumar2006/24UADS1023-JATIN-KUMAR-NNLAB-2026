# 🚗 Driving Behavior Analysis using Deep Learning

## 📌 Project Overview
This project analyzes driving behavior using smartphone sensor data (accelerometer + gyroscope) and predicts a **Driving Score (1–5)**.

The system evaluates how safely a person drives a two-wheeler and classifies behavior into different safety levels.

---

## 🎯 Objectives
- Collect real-world driving data using mobile sensors  
- Build a dataset with multiple riders, vehicles, and conditions  
- Train deep learning models for behavior classification  
- Generate a **Driver Risk / Safety Score**

---

## 🧠 Models Used
- 🔵 LSTM (Long Short-Term Memory)  
- 🟢 GRU (Gated Recurrent Unit)  
- 🟣 Transformer (Attention-based model)  

---

## 📊 Features Engineered
- Acceleration Magnitude (`acc_mag`)  
- Gyroscope Magnitude (`gyro_mag`)  
- Jerk (sudden motion detection)  
- Time-series windowing (sequence modeling)  

---

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📁 Project Structure

<pre>
Driving_Project/
│
├── dataset/ # Raw sensor data (ignored in Git)
├── notebooks/ # Model development notebooks
├── models/ # Saved models (.h5)
├── processed_data/ # Cleaned & processed data
├── results/ # Graphs & evaluation
│
├── predict_driver.py # Final testing script
├── .gitignore
└── README.md
</pre>

---

## 🚀 How to Run

### 1️⃣ Clone Repository

git clone https://github.com/Jatinkumar2006/24UADS1023-JATIN-KUMAR-NNLAB-2026.git

cd Driving_Project


---

### 2️⃣ Run Prediction Script

python predict_driver.py


👉 Enter path of your CSV file when prompted  

---

## 📥 Input Format (CSV)


X_Acc, Y_Acc, Z_Acc, X_Gyro, Y_Gyro, Z_Gyro


---

## 📤 Output Example


LSTM Average Rating: ⭐ 1.95
GRU Average Rating: ⭐ 2.95
Transformer Average Rating: ⭐ 3.32

🏆 FINAL DRIVING SCORE: ⭐ 2.74 / 5

😐 Normal Driver (3⭐)
⚖️ Moderate driving – sometimes safe, sometimes aggressive.


---

## 🏆 Driver Rating System

| Rating | Category   | Description              |
|--------|------------|--------------------------|
| ⭐ 5    | Excellent  | Very safe driving        |
| ⭐ 4    | Good       | Safe and controlled      |
| ⭐ 3    | Normal     | Moderate driving         |
| ⭐ 2    | Rash       | Risky behavior           |
| ⭐ 1    | Dangerous  | Highly unsafe            |

---

## 📈 Model Evaluation
- Accuracy comparison of LSTM, GRU, Transformer  
- Confusion matrix visualization  
- Training performance graphs  

---

## ⚠️ Note
- Large files (datasets, models) are not uploaded due to GitHub limits  
- You can retrain models using provided notebooks  

---

## 🚀 Future Improvements
- Real-time prediction using mobile app  
- Streamlit Web UI  
- GPS integration  
- Driver monitoring dashboard  

---

## 👨‍💻 Author
**Jatin Kumar**  
AI & Data Science Engineering Student  

---

## 💬 Acknowledgment
Thanks to mentors and faculty for guidance in this project  

---

## ⭐ If you like this project
Give it a ⭐ on GitHub!
