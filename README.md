# 🩺 Breast Cancer Prediction App

A Streamlit web app that benchmarks **7 machine learning classifiers**, auto-selects the best one by ROC-AUC, and provides an interactive prediction interface for breast cancer diagnosis.

## 🚀 Live Demo
Deploy on [Streamlit Cloud](https://streamlit.io/cloud) — see deployment steps below.

## 📊 Models Compared
| Model | ROC-AUC |
|---|---|
| 🏆 Logistic Regression | **0.9960** |
| Gradient Boosting | 0.9954 |
| SVM (RBF) | 0.9947 |
| Random Forest | 0.9942 |
| AdaBoost | 0.9871 |
| KNN | 0.9823 |
| Decision Tree | 0.9246 |

## 📁 Project Structure
```
breast-cancer-app/
├── app.py              # Streamlit app (trains model at startup)
├── data.csv            # Wisconsin Breast Cancer Dataset
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## 🛠️ Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-app.git
cd breast-cancer-app
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch `main`, and file `app.py`
5. Click **Deploy** — done!

## 📋 Dataset
Wisconsin Breast Cancer Dataset — 569 samples, 30 features computed from digitized
fine needle aspirate (FNA) images of breast masses.

## ⚠️ Disclaimer
This app is for **educational and research purposes only**. It is not a substitute
for professional medical diagnosis or treatment.
