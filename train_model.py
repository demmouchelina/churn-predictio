import pandas as pd

# Read csv
data = pd.read_csv("churn.csv")

# Display the first few rows of the dataset
print(data.head())

# Describe the dataset
print(data.describe())

# Get information about the dataset
print(data.info())


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Distribution of the target variable
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# 2. Missing values analysis
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values 
data.fillna(data.median(numeric_only=True), inplace=True)

# One-hot encode categorical columns
data_encoded = pd.get_dummies(data)

# Compute the correlation matrix on the encoded dataset
correlation_matrix = data_encoded.corr()

# Plot the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()




# 4. Distribution of numeric features
data.hist(figsize=(10, 8))
plt.title('Distribution of Numeric Features')
plt.show()


import numpy as np

# Check for duplicate values
duplicates = data.duplicated().sum()
print(f"Number of duplicated rows: {duplicates}")

#

# Fill missing values for numeric columns with the median
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing values for categorical columns with the mode
categorical_cols = data.select_dtypes(include=[object]).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Prepare features and target variable
X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

# Ensure the target variable is categorical
y = y.astype(int)




# Feature scaling 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# Create a new DataFrame with scaled features
data_scaled = pd.DataFrame(scaled_features, columns=data.select_dtypes(include=[np.number]).columns)

# Split Train and Test dataset
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score



models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"\n{name} - Confusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test))
    print(f"{name} - Accuracy (Test):", accuracy_score(y_test, y_pred_test))
    print(f"{name} - Recall (Test):", recall_score(y_test, y_pred_test))
    print(f"{name} - Precision (Test):", precision_score(y_test, y_pred_test))
    print(f"{name} - F1-Score (Test):", f1_score(y_test, y_pred_test))
    
    # Check for overfitting/underfitting
    print(f"{name} - Accuracy (Train):", accuracy_score(y_train, model.predict(X_train)))

import joblib

best_model = DecisionTreeClassifier()  
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_model.pkl')
