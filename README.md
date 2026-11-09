# 📊 Automatic Customer Complaint Analysis

An intelligent machine-learning-based application that automatically categorizes customer complaints, generates analytical insights, identifies potential new complaint patterns, and produces downloadable reports.

The application is designed to help organizations analyze large volumes of customer complaints efficiently and understand the most frequently reported issues.

## 🚀 Project Overview

Manually categorizing customer complaints can be time-consuming, inconsistent, and difficult to manage when dealing with large datasets.

This project provides an automated solution using Natural Language Processing (NLP) and Machine Learning.

Users can upload an Excel or CSV file containing customer complaints, and the application automatically:

- Predicts complaint categories.
- Calculates category-wise complaint counts and percentages.
- Identifies low-confidence predictions requiring human review.
- Discovers potential new complaint groups using clustering.
- Generates visualizations using Matplotlib.
- Provides downloadable Excel and PDF reports.

The application uses a pre-trained model, so users do not need to provide training data.

## ✨ Key Features

### 1. Automatic Complaint Categorization
- Uses Sentence Transformers for text representation.
- Uses Logistic Regression for complaint classification.
- Automatically predicts categories for uploaded complaints.

### 2. Complaint Analytics Dashboard
- Total complaints processed.
- Number of predicted categories.
- Complaints flagged for review.
- Top three complaint categories and their share.

### 3. Category-wise Visualization
- Generates horizontal bar charts using Matplotlib.
- Displays complaint counts for individual categories.
- Uses different colours for different complaint categories.

### 4. Confidence-Based Human Review
- Identifies predictions that do not meet the configured confidence threshold.
- Flags uncertain predictions for manual verification.

### 5. Potential New Complaint Group Discovery
- Uses clustering to identify similar complaint descriptions.
- Highlights exploratory groups that may represent emerging complaint categories.
- Does not automatically convert these groups into official categories.

### 6. Downloadable Reports
- Excel report containing categorized complaints, category summaries, and candidate groups.
- PDF report containing analytical summaries and visualizations.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and clustering |
| Sentence Transformers | Text embeddings |
| Matplotlib | Data visualization |
| Joblib | Model serialization |
| OpenPyXL | Excel file processing |
| ReportLab | PDF report generation |

## 📁 Project Structure

```text
Automatic-customer-complaint-analysis/
│
├── app.py
│
├── complaint_analyzer/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── modeling.py
│   └── reporting.py
│
├── training/
│   ├── train_model.py
│   └── Consumer Complain.xlsx
│
├── models/
│   └── complaint_model.joblib
│
├── outputs/
│
├── tests/
│   └── test_data_utils.py
│
├── .github/
│   └── CODEOWNERS
│
├── .gitignore
├── requirements.txt
└── README.md
```

Note: The training dataset and generated model file are maintained separately from the public source code repository when confidential or large files should not be shared.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 recommended.
- Git.
- Visual Studio Code (recommended).

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Automatic-customer-complaint-analysis.git
```

Navigate into the project directory:

```bash
cd Automatic-customer-complaint-analysis
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the Trained Model

The application requires a pre-trained model.

The developer must:

1. Place the approved labelled training dataset inside the `training/` directory.
2. Run the model training script.
3. Generate the trained model artifact inside the `models/` directory.

Run:

```bash
python training/train_model.py
```

The expected model file is:

```text
models/complaint_model.joblib
```

Users do not need the original training dataset to analyze complaints, provided the trained model is already available.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## 📈 Application Workflow

```text
User Uploads Excel/CSV File
          |
          v
   Data Processing
          |
          v
   Complaint Text Extraction
          |
          v
   Pre-trained ML Model
          |
          v
   Category Prediction
          |
          v
 Confidence-Based Review
          |
          v
 Candidate Group Discovery
          |
          v
   Analytical Dashboard
          |
          v
 Matplotlib Visualizations
          |
          v
 Excel and PDF Reports
```

## 🔒 Data Privacy and Security

- Training data should remain private and should not be uploaded to a public repository.
- The application does not require a paid external