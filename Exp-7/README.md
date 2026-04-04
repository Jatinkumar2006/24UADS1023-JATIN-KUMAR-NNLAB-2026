# 🫁 Pneumonia Detection using Deep Learning

A deep learning-based web application that classifies chest X-ray images as **Normal** or **Pneumonia** using **Transfer Learning (ResNet18)**.

---

## 📌 Project Overview
This project uses a pretrained **ResNet18 model (ImageNet)** and applies **transfer learning + fine-tuning** to classify medical images.

The model is deployed using **Streamlit**, allowing users to upload X-ray images and get real-time predictions.

---

## 🎯 Objectives
* Apply **transfer learning** on a medical dataset.
* Improve performance using **fine-tuning**.
* Build a **real-world deployable application**.
* Evaluate using multiple performance metrics.

---

## 🧠 Model Details
* **Model:** ResNet18 (Pretrained on ImageNet).
* **Technique:** Transfer Learning + Fine-Tuning.
* **Framework:** PyTorch.
* **Classes:**
  * Normal
  * Pneumonia

---

## 📊 Performance
| Metric | Value |
| :--- | :--- |
| Training Accuracy | ~98.7% |
| Validation Accuracy | ~99.1% |
| **Test Accuracy** | **97.96%** |

---

## 🗂️ Project Structure
```text
Exp-7/
│
├── data/               # Dataset directory
│   └── chest_xray/
│
├── models/             # Saved model weights
│   ├── best_model.pth
│   └── final_model.pth
│
├── outputs/            # Generated plots and metrics
│   └── graphs/
│
├── notebooks/          # Training and experiments
│   └── transfer_learning.ipynb
│
├── app.py              # Streamlit application script
└── README.md           # Project documentation
---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd Exp-7
```

### 2. Create Environment
```bash
conda create -n torch_env python=3.10 -y
conda activate torch_env
```

### 3. Install Dependencies
```bash
pip install torch torchvision streamlit matplotlib seaborn scikit-learn tqdm
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

---

## 🧪 How It Works
1. **Upload** an X-ray image via the web interface.
2. **Preprocessing:** Image is automatically resized and normalized.
3. **Prediction:** The model processes the image and predicts the class.
4. **Output:** Results are displayed with a confidence score.

---

## 💡 Key Learnings
* **Transfer learning** significantly reduces training time.
* **Fine-tuning** improves model accuracy.
* Proper **data splitting** prevents overfitting.
* **Deployment** bridges ML research and real-world applications.

---

## 🚀 Future Improvements
* Add **Grad-CAM heatmaps** for better explainability.
* Deploy the application online using **Streamlit Cloud**.
* Experiment with deeper architectures like **ResNet50**.
* Improve **dataset diversity** to enhance generalization.

---

## 💬 Conclusion
This project demonstrates that transfer learning is highly effective for medical image classification, achieving high accuracy even with limited data.

---

## 👨‍💻 Author
**Jatin Kumar**
*AI & Data Science Engineering Student*

---

## 🙏 Acknowledgements
This project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

I sincerely thank the dataset creators and contributors for making this valuable medical dataset publicly available, which enabled the development and evaluation of this deep learning model.

Dataset is not included due to size limitations.
Download from below link and Place it inside: <b>data/chest_xray/</b>

🔗 **Dataset Link:** [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) ```