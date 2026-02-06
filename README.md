# ❤️ Heart Sound Classification using Deep Neural Networks

## 📌 Project Overview

This project focuses on building a Machine Learning model capable of
classifying heart sound audio recordings into two categories using Deep
Neural Networks (DNN). The system processes raw heart sound audio files,
extracts acoustic features using MFCC (Mel-Frequency Cepstral
Coefficients), and trains a neural network to perform binary
classification.

The project is inspired by datasets used in clinical audio challenges
such as the PhysioNet / Computing in Cardiology Challenge 2016.

------------------------------------------------------------------------

## 🎯 Objective

-   Develop a robust ML model for classifying heart sound recordings\
-   Achieve high performance using:
    -   Accuracy\
    -   Precision\
    -   Recall\
    -   F1 Score

------------------------------------------------------------------------

## 📂 Dataset Information

-   Total Samples: 3240 audio recordings\
-   Format: `.wav` audio files

Labels: - 1 → Positive Class\
- -1 → Negative Class (Converted to 1 and 0 for training)

Data Structure: training-a/\
training-b/\
training-c/\
training-d/\
training-e/\
training-f/\
REFERENCE.csv

------------------------------------------------------------------------

## 🧠 Model Architecture

Deep Neural Network (Multi-Layer Perceptron)

-   Input Layer: 40000 features\
-   Dense Layer: 256 (ReLU)\
-   Dropout\
-   Dense Layer: 128 (ReLU)\
-   Dropout\
-   Output Layer: 2 (Softmax)

Total Parameters: \~10.2 Million\
Optimizer: Adam\
Loss Function: Categorical Crossentropy

------------------------------------------------------------------------

## ⚙️ Preprocessing Pipeline

### Feature Extraction

-   MFCC Features using librosa\
-   Number of MFCCs: 40

### Audio Processing

-   Resampled to 22050 Hz\
-   Fixed length: 1000 time frames\
-   Padding → Short audio\
-   Truncation → Long audio

### Feature Transformation

-   MFCC Matrix: 40 × 1000\
-   Flattened to 40000 feature vector

### Data Preparation

-   Label Encoding\
-   Train-Test Split → 80 / 20\
-   Feature Scaling → StandardScaler

------------------------------------------------------------------------

## 📊 Results

-   Test Accuracy: 91.66%\
-   High Recall for Positive Class: 0.97

------------------------------------------------------------------------

## 🚀 Technologies Used

-   Python\
-   TensorFlow / Keras\
-   NumPy\
-   Pandas\
-   Scikit-learn\
-   Librosa\
-   Matplotlib

------------------------------------------------------------------------

## 📦 Installation

git clone https://github.com/your-username/your-repo-name.git\
cd your-repo-name\
pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ How to Run

python train_model.py

Or run Jupyter Notebook: jupyter notebook

------------------------------------------------------------------------

## 📈 Future Improvements

-   Implement CNN / RNN / LSTM architectures\
-   Add more audio features (Chroma, Spectral Contrast)\
-   Hyperparameter optimization\
-   Deploy as web or clinical support tool

------------------------------------------------------------------------

## 📜 License

Educational and research purposes only.

------------------------------------------------------------------------

## 👨‍💻 Author

Vishal Tayade
