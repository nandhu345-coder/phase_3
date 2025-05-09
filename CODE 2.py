# Phase-2: Predicting Customer Churn using Machine Learning
# Author: Mohammed Aasif

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Step 2: Create a Larger Sample Dataset with equal-length columns
n_samples = 20

# Cycle contract values safely
contract_values = ['Month-to-month', 'One year', 'Two year']
contract_column = [contract_values[i % 3] for i in range(n_samples)]

data = {
    'customerID': [f'{i:03}' for i in range(1, n_samples + 1)],
    'gender': ['Female', 'Male'] * (n_samples // 2),
    'SeniorCitizen': [0, 1] * (n_samples // 2),
    'Partner': ['Yes', 'No'] * (n_samples // 2),
    'Dependents': ['No', 'Yes'] * (n_samples // 2),
    'tenure': np.random.randint(1, 72, n_samples),
    'PhoneService': ['Yes', 'No'] * (n_samples // 2),
    'InternetService': ['DSL', 'Fiber optic'] * (n_samples // 2),
    'Contract': contract_column,
    'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
    'TotalCharges': np.round(np.random.uniform(100, 5000, n_samples), 2),
    'Churn': ['No', 'Yes'] * (n_samples // 2)
}
df = pd.DataFrame(data)

# Step 3: Preprocessing
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'Churn']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 4: Feature Engineering
df['TotalServicesUsed'] = df['PhoneService'] + df['InternetService']
df['EngagementScore'] = df['Contract'] * df['tenure']

# Step 5: Feature Selection
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Step 6: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Step 8: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Step 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()