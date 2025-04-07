import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets["quality"]

y_grouped = y.copy()
y_grouped[y <= 4] = 0
y_grouped[(y > 4) & (y <= 6)] = 1
y_grouped[y > 6] = 2

print("First 5 rows of the dataset:")
print(pd.concat([X, y], axis=1).head())

stats_df = pd.DataFrame({
    "Mean": X.mean(),
    "Variance": X.var(),
    "Skewness": X.skew(),
    "Entropy": X.apply(lambda x: entropy(x.value_counts()))
})

print("\nStatistical properties of each feature:")
print(stats_df)

combined = pd.concat([X, y], axis=1)
plt.figure(figsize=(12, 8))
sns.heatmap(combined.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

combined.hist(figsize=(12,10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize= 16)
plt.show()

high_variance_col = stats_df["Variance"].idxmax()
high_entropy_col = stats_df["Entropy"].idxmax()

print(f"\nFeature with highest variance: {high_variance_col}")
print(f"Feature with highest entropy: {high_entropy_col}")

X_drop_variance = X.drop(columns=[high_variance_col])
X_drop_entropy = X.drop(columns=[high_entropy_col])

X_train, X_test, y_train, y_test = train_test_split(X, y_grouped, test_size=0.2, random_state=42)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_drop_variance, y_grouped, test_size=0.2, random_state=42)
Xe_train, Xe_test, ye_train, ye_test = train_test_split(X_drop_entropy, y_grouped, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_v = RandomForestClassifier()
rf_v.fit(Xv_train, yv_train)

rf_e = RandomForestClassifier()
rf_e.fit(Xe_train, ye_train)

print("\nModel Evaluation (Original Data):")
print(classification_report(y_test, rf.predict(X_test)))

print("\nModel Evaluation (Dropped High Variance Feature):")
print(classification_report(yv_test, rf_v.predict(Xv_test)))

print("\nModel Evaluation (Dropped High Entropy Feature):")
print(classification_report(ye_test, rf_e.predict(Xe_test)))

print("\nAccuracies:")
print("Original:", accuracy_score(y_test, rf.predict(X_test)))
print("Dropped High Variance:", accuracy_score(yv_test, rf_v.predict(Xv_test)))
print("Dropped High Entropy:", accuracy_score(ye_test, rf_e.predict(Xe_test)))
