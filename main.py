import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv('/Users/mo/Downloads/vaccine_prediction.csv')
print("First 5 rows of the dataset:\n", df.head())
print(df.describe())
# Data preprocessing
df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0})
df['race'] = df['race'].replace({'White': 0, 'Black': 1, 'Other or Multiple': 2, 'Hispanic': 3})
df['qualification'] = df['qualification'].replace({'< 12 Years': 0, '12 Years': 1, 'College Graduate': 2, 'Some College': 3})
df['age_bracket'] = df['age_bracket'].replace({'18 - 34 Years': 0, '35 - 44 Years': 1, '45 - 54 Years': 2, '55 - 64 Years': 3, '65+ Years': 4})
df['marital_status'] = df['marital_status'].replace({'Not Married': 0, 'Married': 1})
df['income_level'] = df['income_level'].replace({'<= $75,000, Above Poverty': 0, '> $75,000': 1, 'Below Poverty': 2})
df['housing_status'] = df['housing_status'].replace({'Own': 1, 'Rent': 0})
df['census_msa'] = df['census_msa'].replace({'Non-MSA': 0, 'MSA, Not Principle  City': 1, 'MSA, Principle City': 2})
df['employment'] = df['employment'].replace({'Unemployed': 0, 'Employed': 1, 'Not in Labor Force': 2})

# Drop irrelevant columns
columns_to_drop = ['unique_id', 'race', 'sex', 'no_of_children']
df = df.drop(columns=columns_to_drop)
print("First 5 rows after dropping columns:\n", df.head())

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and target variable (y)
X = df_imputed.drop('h1n1_vaccine', axis=1)
y = df_imputed['h1n1_vaccine']

# Create dataset
X, y = make_classification(n_samples=26706, n_features=30, n_informative=4, n_redundant=15, random_state=1)
# Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Summarize the shape of the train and test sets
print("Train and test set shapes:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize logistic regression model
logistic_reg_mle_model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Fit the model to the training data
logistic_reg_mle_model.fit(X_train, y_train)

# Predict on the test data
y_pred_mle = logistic_reg_mle_model.predict(X_test)

# Evaluate the model
accuracy_mle = accuracy_score(y_test, y_pred_mle)
precision_mle = precision_score(y_test, y_pred_mle)
recall_mle = recall_score(y_test, y_pred_mle)
f1_mle = f1_score(y_test, y_pred_mle)
conf_matrix_mle = confusion_matrix(y_test, y_pred_mle)

print("Logistic Regression Model with MLE:")
print("Accuracy:", accuracy_mle)
print("Precision:", precision_mle)
print("Recall:", recall_mle)
print("F1 Score:", f1_mle)
print("Confusion Matrix:\n", conf_matrix_mle)

# Create a confusion matrix dataframe
lg = pd.DataFrame(index=['Actual Negative', 'Actual Positive'],
                  columns=['Predicted Negative', 'Predicted Positive'])
lg.loc['Actual Negative', 'Predicted Negative'] = conf_matrix_mle[0, 0]
lg.loc['Actual Negative', 'Predicted Positive'] = conf_matrix_mle[0, 1]
lg.loc['Actual Positive', 'Predicted Negative'] = conf_matrix_mle[1, 0]
lg.loc['Actual Positive', 'Predicted Positive'] = conf_matrix_mle[1, 1]

print("Confusion Matrix:")
print(lg)

# Predict on the training data
y_train_pred_mle = logistic_reg_mle_model.predict(X_train)

# Evaluate the model on the training data
accuracy_train_mle = accuracy_score(y_train, y_train_pred_mle)
precision_train_mle = precision_score(y_train, y_train_pred_mle)
recall_train_mle = recall_score(y_train, y_train_pred_mle)
f1_train_mle = f1_score(y_train, y_train_pred_mle)
conf_matrix_train_mle = confusion_matrix(y_train, y_train_pred_mle)

print("Performance on Training Data:")
print("Accuracy:", accuracy_train_mle)
print("Precision:", precision_train_mle)
print("Recall:", recall_train_mle)
print("F1 Score:", f1_train_mle)
print("Confusion Matrix:")
print(conf_matrix_train_mle)

# Plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves
title = "Learning Curves (Logistic Regression)"
plot_learning_curve(logistic_reg_mle_model, title, X_train, y_train, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()

# Check for overfitting or underfitting
def check_overfitting_underfitting(train_score, test_score, threshold=0.05):
    if train_score > test_score + threshold:
        return "Overfitting"
    elif train_score < test_score - threshold:
        return "Underfitting"
    else:
        return "Good Fit"

# Calculate train and test scores
train_score = accuracy_train_mle
test_score = accuracy_mle

# Check for overfitting or underfitting
result = check_overfitting_underfitting(train_score, test_score)
print("Model Evaluation:", result)
