# PCOS Early Detection System

## 1. Project Overview

PCOS Early Detection System is a Machine Learning-based project that predicts the possibility of Polycystic Ovary Syndrome (PCOS) using patient health-related information. It analyzes factors such as age, BMI, menstrual cycle patterns, hormonal levels, and ultrasound results.

The system uses Random Forest and Decision Tree classification algorithms to predict PCOS status and compares their performance using different evaluation metrics. It supports early risk identification but does not replace professional medical diagnosis.

## 2. Objectives

- To predict PCOS status using patient health-related features.
- To apply machine learning classification techniques.
- To compare Random Forest and Decision Tree algorithms.
- To evaluate model performance using Accuracy, Precision, Recall, and F1-Score.
- To identify the better-performing algorithm.

## 3. Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Data Preprocessing:** One-Hot Encoding
- **Algorithms:** Random Forest Classifier, Decision Tree Classifier
- **Dataset Format:** CSV (.csv)

## 4. Features

- Predicts PCOS status based on patient health information.
- Analyzes numerical and categorical health features.
- Handles missing values during data preprocessing.
- Converts categorical data into numerical values.
- Compares two machine learning classification algorithms.
- Evaluates model performance using four metrics.
- Displays prediction results as Yes or No.
- Identifies the better-performing model.

## 5. Inputs

The system uses the following patient-related inputs:

- Age
- Weight
- Height
- Body Mass Index (BMI)
- Menstrual cycle information
- LH hormone level
- FSH hormone level
- Testosterone level
- Ultrasound results

These inputs are used by the trained machine learning models to predict PCOS status.

## 6. Methodology

1. **Data Collection:** Load the PCOS dataset.
2. **Data Preprocessing:** Clean the dataset and handle missing values.
3. **Feature Encoding:** Apply One-Hot Encoding to convert categorical features into numerical values.
4. **Data Splitting:** Divide the dataset into 80% training data and 20% testing data.
5. **Model Training:** Train Random Forest and Decision Tree classification algorithms.
6. **Model Evaluation:** Evaluate both models using Accuracy, Precision, Recall, and F1-Score.
7. **Model Comparison:** Compare the evaluation results and identify the best-performing model.
8. **Prediction:** Predict PCOS status based on patient health details.

## 7. Working Process

1. Load the PCOS dataset.
2. Preprocess the numerical and categorical data.
3. Handle missing values and encode categorical features.
4. Split the dataset into training and testing sets.
5. Train Random Forest and Decision Tree models.
6. Evaluate and compare both algorithms.
7. Select the better-performing model.
8. Enter patient health details and generate the predicted PCOS status.

## 8. Algorithms Used

### Random Forest Classifier

Random Forest is a supervised machine learning classification algorithm that combines multiple decision trees to produce a prediction. It uses the combined results of multiple trees to classify PCOS status.

### Decision Tree Classifier

Decision Tree is a supervised machine learning classification algorithm that uses a tree-like structure to make predictions based on different patient health features.

### Algorithm Comparison

| Evaluation Metric | Random Forest | Decision Tree |
|---|---:|---:|
| Accuracy | 98.50% | 97.00% |
| Precision | 0.9789 | 0.9738 |
| Recall | 0.9886 | 0.9594 |
| F1-Score | 0.9835 | 0.9661 |

**Best Performing Algorithm:** Random Forest Classifier

Random Forest achieved better performance across all four evaluation metrics compared to Decision Tree.

## 9. Dataset Description

**Dataset Name:** `pcos_dataset.csv`

**Total Records:** 1,000

**Total Features:** 16

**Target Column:** `PCOS_Status`

### Dataset Features

The dataset contains patient health-related numerical and categorical attributes, including:

- Age
- Weight
- Height
- BMI
- Menstrual cycle information
- Hormonal levels
- Ultrasound-related information
- Other available health-related features

### Output Classes

- **Yes:** PCOS is predicted based on the provided patient details.
- **No:** PCOS is not predicted based on the provided patient details.

### Dataset Processing

- Loaded the dataset using Pandas.
- Handled missing values in numerical and categorical features.
- Applied One-Hot Encoding to categorical features.
- Divided the dataset into 80% training and 20% testing data.
- Used the processed data to train and evaluate both algorithms.

## 10. Future Enhancements

- Improve prediction performance using larger and more diverse datasets.
- Develop a user-friendly web application.
- Add graphical reports for patient health information.
- Include additional relevant health-related features.
- Explore advanced machine learning techniques.
- Provide more detailed prediction results to support early risk identification.

## 11. Conclusion

The PCOS Early Detection System uses Machine Learning algorithms to predict PCOS status based on patient health-related features. By comparing Random Forest and Decision Tree, Random Forest achieved better performance with 98.50% accuracy and a 0.9835 F1-score. This project demonstrates how machine learning can support early risk identification.

**Note:** This project is developed for educational purposes and does not replace professional medical diagnosis.

## 12. Author

**Namitha**

B.Tech – Artificial Intelligence and Data Science

The Apollo University

GitHub: [Add your GitHub Profile Link]
