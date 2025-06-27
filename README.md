
# Logistic Regression - Diabetes Classification

## Task Overview
**Internship Task 4: Classification with Logistic Regression**  
This task involves building a binary classification model using logistic regression to predict whether a patient has diabetes based on health-related features.

## Objective
To apply logistic regression for solving a binary classification problem using a real-world dataset and evaluate the model using appropriate metrics.

## Dataset Information
- **Name**: Pima Indians Diabetes Dataset  
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Format**: CSV  
- **Target Variable**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

The dataset includes medical predictor variables such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Project Structure
```
.
├── logistic_regression_diabetes.ipynb   # Jupyter notebook with full implementation
├── pima-indians-diabetes.data.csv       # Dataset 
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies

```

## Tools and Libraries
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Implementation Steps

### 1. Import Libraries
Essential Python libraries and modules are imported for data manipulation, visualization, model training, and evaluation.

### 2. Load Dataset
The dataset is loaded from a public URL using `pandas`. Column names are manually assigned based on the dataset's documentation.

### 3. Data Inspection
Basic checks are performed to understand the structure, size, and presence of missing values.

### 4. Train-Test Split
The data is split into training and testing sets using `train_test_split`, ensuring the class distribution is preserved (`stratify=y`).

### 5. Feature Scaling
Standardization of features is done using `StandardScaler` to ensure consistent model performance.

### 6. Model Training
A logistic regression model is trained on the scaled training data.

### 7. Model Prediction
Predictions and prediction probabilities are generated on the test data.

### 8. Evaluation Metrics
The following evaluation metrics are used:
- Confusion Matrix
- Classification Report
- Precision
- Recall
- ROC-AUC Score

### 9. ROC Curve
The ROC curve is plotted to visualize the model's performance across different thresholds.

### 10. Threshold Tuning
The classification threshold is adjusted manually (e.g., to 0.3) to observe changes in precision and recall.

### 11. Sigmoid Function
A plot of the sigmoid function is included to explain how logistic regression outputs probabilities.

## Learning Outcomes
- Understanding of logistic regression for binary classification
- Data preprocessing (scaling, splitting)
- Evaluation of classification models using industry-standard metrics
- Role of the sigmoid function in logistic regression
- Importance of classification thresholds in real-world problems

## How to Run

1. **Clone this repository** or download the files.
2. Open `logistic_regression_diabetes.ipynb` using:
   - Jupyter Notebook
   - Google Colab *(Replace `github.com` with `githubtocolab.com` in the notebook URL if Colab preview is not available.)*
3. Run each cell sequentially to reproduce the results.

## Requirements

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

### `requirements.txt` content:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```


