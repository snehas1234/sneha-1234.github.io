import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('breast_cancer_bd.csv', na_values='?')
df = df.dropna()

print("\nSample rows (before transformation):")
print(df.sample(5))

print("\nUnique values per column:")
print(df.nunique())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDataFrame info:")
print(df.info())

df = df.drop(columns=['Sample code number'])

df.drop(columns=['Class']).hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

df = df.drop(columns=['Bare Nuclei'])


corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation")
plt.show()

df['Class'] = df['Class'].map({2: 0, 4: 1})

print("\nSample rows (after class mapping):")
print(df.sample(5))

x = df.drop(columns=['Class'])
y = df['Class']

st_scaler = StandardScaler()
scaled_x = st_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

svm = SVC()
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name}\n")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_svm, "Support Vector Machine")
