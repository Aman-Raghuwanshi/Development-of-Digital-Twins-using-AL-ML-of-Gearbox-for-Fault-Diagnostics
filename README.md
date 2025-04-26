# 🚀 Gearbox Fault Detection using AI/ML

This is a Streamlit-based web app that predicts whether a **gearbox system is faulty** using Machine Learning.

🔗 **Live Demo**: [gearboxfaultdetection.streamlit.io](https://gearboxfaultdetection.streamlit.io)

## 💡 Project Overview

- Built as part of a Bachelor's thesis project titled **"Development of Digital Twins using AI/ML for Gearbox Fault Diagnostics"**
- Models used: Support Vector Machine (SVM) and Logistic Regression
- Data preprocessing includes:
  - Imputation (missing values)
  - Feature Scaling
- ML pipeline built using `scikit-learn`, deployed with `Streamlit`

## 📁 Folder Structure

```
.
├── README.md
├── app.py                 # Streamlit frontend
├── model.py               # Model training and saving
├── data1_gear.csv         # Gearbox dataset
├── requirements.txt       # Python dependencies
└── .gitignore
```

## ⚙️ How to Run Locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
python3 -m venv venv
source venv/bin/activate    # On Windows use venv\Scripts\activate
pip install -r requirements.txt

# (Optional) Train models
python model.py

# Run the Streamlit app
streamlit run app.py
```

## 📌 Input Features

- No. of Teeth on Gear
- No. of Teeth on Pinion
- Average Gear Ratio
- Total Deformation
- Equivalent Stress
- Maximum Principal Elastic Strain
- Chip Length

## 🧠 Prediction Output

The app shows whether the gearbox is **FAULTY** or **NOT FAULTY** using the selected model.

## 🤝 Contributions

Built by Aman Raghuwanshi at IIT Guwahati as part of academic research.

---
