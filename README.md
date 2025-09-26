# task-4

Objective
Build a binary classification model using Logistic Regression to predict whether a case is malignant or benign, following the Elevate AI ML INTERNSHIP Labs assignment.

Dataset
data.csv: Contains features extracted from breast cancer cell samples.

Target variable: diagnosis
M = Malignant (encoded as 1)
B = Benign (encoded as 0)

Excluded: iddiagnosis column (acts as an index).

Tools Used
Python 3.10+
pandas
scikit-learn
matplotlib

Steps Followed
1. Data Preprocessing
Loaded the dataset using pandas.
Encoded the target variable ('diagnosis': M=1, B=0).
Removed the iddiagnosis column and retained only relevant feature columns.
Checked for missing values and ensured data cleanliness.

2. Train-Test Split & Feature Scaling
Split the data into training and test sets (80/20 split) using train_test_split.
Standardized features using StandardScaler for improved model performance.

3. Model Training
Used scikit-learn's LogisticRegression to fit a binary classifier on the training set.

4. Evaluation
Evaluated performance on the test set using:
Confusion Matrix: Visual diagnosis of TP, TN, FP, FN.
Precision, Recall, F1-Score: Quantitative assessment of performance.
ROC Curve and AUC: Ability to distinguish classes at various thresholds.
Experimented with different classification thresholds and discussed their effects.

5. Results
Insert summary table or description of:
Confusion matrix outcome
Precision, Recall, F1, ROC-AUC values
Plotted ROC curve for visual reference.

Brief Explanations & Interview Questions
1. How does logistic regression differ from linear regression?
Logistic regression predicts probabilities using a sigmoid function and is used for classification, while linear regression predicts continuous numeric values.

2. What is the sigmoid function?
The sigmoid function maps real-valued numbers to the (0,1) interval: σ(z)=1+e−z
This output is interpreted as the probability of the sample belonging to the positive class.

3. What is precision vs recall?
Precision: Proportion of correct positive predictions (TP / (TP+FP)).
Recall: Proportion of actual positives correctly predicted (TP / (TP+FN)).

4. What is the ROC-AUC curve?
ROC curve: Plots TPR vs FPR at various thresholds.
AUC (Area Under Curve): Measures model's ability to discriminate between classes.

5. What is the confusion matrix?
A confusion matrix is a summary table showing correct and incorrect predictions, structured as TP, TN, FP, and FN.

6. What happens if classes are imbalanced?
If one class dominates, accuracy can be misleading. Precision, recall, and F1-score become more informative. Techniques like resampling or adjusting class weights may be required.

7. How do you choose the threshold?
The threshold balances precision and recall. You may select it based on the ROC curve, or to match the cost/risk of false positives and negatives in your application.

8. Can logistic regression be used for multi-class problems?
Yes, using extensions such as One-vs-Rest (OvR) or multinomial logistic regression with a softmax function.

Files in this Repository
data.csv – Input dataset.
task4_logistic_regression.ipynb or .py – Code for all steps.
plots/ – Confusion matrix, ROC curves, etc. (if any)

README.md – This documentation.

Instructions
Clone this repo.
Run the notebook or Python script to reproduce the analysis.
Review results and plots; customize as needed for your own use case.

References
Scikit-learn Documentation
Matplotlib Documentation

Dataset: UCI Machine Learning Repository – Breast Cancer Wisconsin Dataset

