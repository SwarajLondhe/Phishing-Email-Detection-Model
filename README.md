# 🎣 Phishing Email Detection Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Overview
This is a Machine Learning project built to classify emails as either **"Phishing"** (malicious/scam) or **"Safe"** (legitimate). It uses Natural Language Processing (NLP) to analyze the textual content of emails and explicitly extracts URL features to detect common phishing patterns.

**Repository Link:** [https://github.com/SwarajLondhe/Phishing-Email-Detection-Model](https://github.com/SwarajLondhe/Phishing-Email-Detection-Model)

## ⚙️ Key Features
* **URL Feature Extraction:** Uses Regular Expressions (Regex) to automatically hunt down hidden web links and tag them as a primary suspicious feature.
* **Keyword Analysis:** Utilizes `TfidfVectorizer` to convert email text into numerical data, focusing on high-priority words (e.g., "urgent", "password", "bank").
* **Machine Learning Pipeline:** Employs a **Logistic Regression** classifier to deliver high-speed, accurate predictions while preventing data leakage.
* **Detailed Evaluation:** Automatically generates Model Accuracy, a Confusion Matrix, and a comprehensive Classification Report.

## 🛠️ Technologies Used
* **Python**
* **Scikit-learn** (Machine Learning & Pipelines)
* **Pandas** (Data structure and matrix formatting)
* **NumPy** (Numerical operations)
* **Regex** (Text preprocessing and URL detection)

## 🚀 How to Install and Run

**1. Clone this repository to your local machine:**
```bash
git clone https://github.com/SwarajLondhe/Phishing-Email-Detection-Model.git
