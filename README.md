# 🎧 Streaming Intelligence Platform  
## AI-Powered Customer Churn Prediction Dashboard

A production-style machine learning web application designed to predict customer churn for subscription-based streaming platforms (e.g., Spotify, Netflix).

This system combines predictive analytics, business intelligence, and a professional dashboard interface to help organizations identify high-risk customers and take proactive retention measures.

---

## 🚀 Executive Summary

Customer churn directly impacts recurring revenue in subscription businesses.  
This project delivers an end-to-end churn prediction solution that:

- Trains a machine learning model on behavioral data  
- Evaluates model performance using industry metrics  
- Deploys a real-time prediction dashboard  
- Provides business-friendly churn insights  
- Uses a modern, Spotify-inspired analytics UI  

The application demonstrates practical integration of Machine Learning + Web Deployment.

---

## 🧠 Machine Learning Architecture

### Model Objective
Binary Classification:
- `0` → Customer Retained  
- `1` → Customer Churn  

### Feature Engineering

The model uses the following behavioral indicators:

| Feature | Description |
|----------|-------------|
| Subscription Months | Tenure duration |
| Monthly Fee | Plan pricing tier |
| Watch Time (Last 30 Days) | Engagement metric |
| Login Frequency | Usage intensity |
| Complaints | Customer dissatisfaction |
| Payment Failure | Billing risk indicator |
| Subscription Type | Tier classification |

---

### Model Development Process

1. Synthetic dataset generation (2000 users)
2. Train-test split (80/20)
3. Model comparison:
   - Logistic Regression
   - Random Forest Classifier
4. Evaluation metrics:
   - Accuracy
   - ROC-AUC Score
5. Random Forest selected for superior performance
6. Model serialized using `joblib`

---

## 📊 Performance Metrics

- Logistic Regression: ~81% Accuracy  
- Random Forest: ~87% ROC-AUC  
- Final Model: Random Forest  

The ROC-AUC metric was prioritized due to probabilistic churn modeling requirements.

---

## 🖥 Application Features

### 🎨 Enterprise-Style UI
- Dark analytics theme
- Spotify-inspired design language
- Clean sidebar input panel
- Professional risk visualization card
- Color-coded churn categories

### 📈 Real-Time Prediction
- Dynamic churn probability calculation
- Risk score progress bar
- High / Medium / Low classification
- Business-friendly churn indicators

### 📊 Analytics Overview
- Total user count
- Overall churn rate

---

## 🛠 Technology Stack

### Backend & ML
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier

### Deployment Layer
- Streamlit
- Matplotlib
- Joblib

---

## 📂 Project Architecture
