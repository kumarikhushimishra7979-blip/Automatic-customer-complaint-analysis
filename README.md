# Automatic-customer-complaint-analysis
An ML-powered customer complaint analyzer built with Python, Streamlit, and NLP. Automatically classifies complaints, discovers potential new complaint groups, visualizes trends, and generates Excel and PDF reports. Designed for local processing, privacy, and easy model retraining with additional datasets.
# Customer Complaint Analyzer 📊

An ML-powered application designed to automate customer complaint categorization and analysis using Natural Language Processing (NLP) and machine learning.

The application processes complaint datasets locally, predicts complaint categories, identifies potential groups of unusual complaints for further investigation, visualizes category distributions, and generates downloadable Excel workbooks and PDF reports.

## ✨ Key Features

* **Automatic Complaint Categorization:** Classifies customer complaints using a TF-IDF and Linear Support Vector Classifier pipeline.
* **Model Training:** Supports training and retraining with multiple labeled CSV and Excel datasets.
* **Category Analytics:** Summarizes complaint volumes and visualizes category distributions.
* **Potential New-Group Discovery:** Uses text clustering to identify candidate groups of lower-confidence complaints for review.
* **Excel Export:** Generates categorized complaint data, category summaries, and candidate-group records.
* **PDF Reporting:** Produces downloadable complaint analysis reports.
* **Local Processing:** Designed to run on a local computer without paid cloud hosting or external AI APIs.
* **Interactive Interface:** Provides a browser-based interface powered by Streamlit.

## 🛠️ Technology Stack

* **Language:** Python
* **Interface:** Streamlit
* **Machine Learning:** Scikit-learn
* **Text Processing:** TF-IDF
* **Classification:** Linear Support Vector Classifier
* **Clustering:** K-Means
* **Data Processing:** Pandas
* **Excel Export:** OpenPyXL
* **Visualization:** Matplotlib
* **PDF Generation:** ReportLab

## 🎯 Project Objective

The goal is to reduce manual effort in analyzing customer complaints, identify recurring issues, support complaint-volume analysis, and help investigate complaint patterns that may require new categories.

## ⚠️ Current Limitations

This project is an evolving prototype. Prediction quality depends on the accuracy, consistency, and coverage of the labeled training data. Model confidence indicators are not calibrated probabilities, and automatically discovered groups are exploratory candidates rather than confirmed new categories. Further evaluation and improvements are required before operational deployment.
