# loan-default-prediction

Machine learning analysis comparing Logistic Regression and Random Forest for predicting loan default risk.

# Loan Default Prediction & Credit Risk Modeling

## Project Overview

This project explores whether borrower and loan characteristics can be used to predict loan default. The analysis compares Logistic Regression with Random Forest to determine whether a more flexible machine learning model provides meaningful improvement over a simpler baseline.

The project also examines an important modeling concern: some variables, such as loan grade and interest rate, may already contain information from a lender's previous risk assessment. A secondary model was created without these variables to evaluate how much predictive performance depended on information that may already reflect prior lending decisions.

## Business Question

Can borrower and loan characteristics be used to identify loans with a higher risk of default, and does Random Forest provide better predictive performance than Logistic Regression?

## Dataset

The Credit Risk Dataset contains borrower and loan information including:

- Age
- Annual income
- Employment length
- Home ownership
- Loan amount
- Interest rate
- Loan intent
- Loan grade
- Loan percent of income
- Previous default history
- Credit history length
- Loan status

The original dataset contained 32,581 observations. After data cleaning, 32,409 observations remained.

Approximately 21.9% of the loans in the dataset were classified as defaults.

## Tools & Technologies

- Python
- pandas
- scikit-learn
- Jupyter Notebook
- Logistic Regression
- Random Forest
- GridSearchCV
- Matplotlib / data visualization libraries

## Methodology

The project followed a complete machine learning workflow:

1. Explored the dataset and identified missing, duplicate, and unrealistic values.
2. Separated the target variable from the predictors.
3. Created an 80/20 stratified training and testing split.
4. Built preprocessing pipelines using:
   - Median imputation for numerical variables
   - Most-frequent imputation for categorical variables
   - Standardization
   - One-hot encoding
5. Trained Logistic Regression as a baseline model.
6. Trained and tuned a Random Forest classifier using GridSearchCV.
7. Evaluated both models using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices.
8. Examined Random Forest feature importance.
9. Trained an additional Random Forest model without loan grade and interest rate to investigate their influence on model performance.

All preprocessing was fit only on the training data to reduce the risk of data leakage.

## Key Results

The tuned Random Forest produced the strongest overall performance:

- Accuracy: 93.47%
- Precision: 96.37%
- Recall: 72.92%
- ROC-AUC: 93.54%
- Cross-validation ROC-AUC: 93.06%

Logistic Regression produced a recall of 55.64%, compared with 72.92% for Random Forest.

Loan percent of income was the strongest feature in the Random Forest model. Interest rate and loan grade also contributed substantial predictive information.

When loan grade and interest rate were removed, recall fell to 52.19%, showing that some of the full model's predictive strength depended on variables that may already incorporate previous lender risk assessments.

## Key Takeaways

The project demonstrates that borrower and loan characteristics can provide useful information for identifying default risk. However, strong predictive performance should be interpreted carefully when some predictors may already contain prior risk judgments.

A model like this would be more appropriate as a decision-support tool than as the sole basis for lending decisions.

## Skills Demonstrated

- Data cleaning and preprocessing
- Exploratory data analysis
- Classification
- Machine learning pipelines
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Feature importance
- Data leakage prevention
- Responsible model interpretation
