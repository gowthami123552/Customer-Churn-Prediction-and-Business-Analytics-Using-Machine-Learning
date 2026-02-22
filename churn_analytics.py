# ============================================
# CUSTOMER CHURN - END TO END PROJECT
# Business Analysis + EDA + SQL + Modeling
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

# --------------------------------------------
# 2. Data Cleaning
# --------------------------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# --------------------------------------------
# 3. Business KPI Analysis (SQL Style)
# --------------------------------------------

total_customers = len(df)
churned_customers = df[df["Churn"] == "Yes"].shape[0]
churn_rate = (churned_customers / total_customers) * 100

print("\n=== Business KPIs ===")
print(f"Total Customers      : {total_customers}")
print(f"Churned Customers    : {churned_customers}")
print(f"Churn Rate (%)       : {churn_rate:.2f}")

print("\nChurn by Contract Type")
print(df.groupby("Contract")["Churn"].value_counts())

print("\nAverage Monthly Charges by Churn")
print(df.groupby("Churn")["MonthlyCharges"].mean())

# --------------------------------------------
# 4. EDA
# --------------------------------------------

plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.xticks(rotation=45)
plt.title("Contract Type vs Churn")
plt.show()

# --------------------------------------------
# 5. Data Encoding for Modeling
# --------------------------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# --------------------------------------------
# 6. Train Test Split
# --------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 7. Feature Scaling
# --------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------
# 8. Logistic Regression Model
# --------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------
# 9. Predictions
# --------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------
# 10. Evaluation Metrics
# --------------------------------------------
print("\n=== Model Performance ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------
# 12. ROC Curve
# --------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# --------------------------------------------
# 13. Business Interpretation
# --------------------------------------------
print("\n=== Business Interpretation ===")
print("Customers with month-to-month contracts and high monthly charges show higher churn probability.")
print("Long tenure customers are less likely to churn.")
print("Retention strategies should target high-risk segments.")
