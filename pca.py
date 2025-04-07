import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('wineds.csv')

print("\nSample rows (before transformation):")
print(df.sample(5))

print("\nUnique values per column:")
print(df.nunique())

print("\nMissing values per column:")
print(df.isnull().sum())

df.drop(columns=['Id'], inplace=True)

print("\nDescriptive statistics:")
print(df.describe())

df.drop(columns=['quality']).hist(figsize=(12,10), bins=20, edgecolor='black')
plt.suptitle("Histogram of Features", fontsize=16)
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

x= df.drop(columns=['quality'])
y= df['quality']

st_scaler= StandardScaler()
scaled= st_scaler.fit_transform(x)

pca= PCA(n_components=2)
pca_data= pca.fit_transform(scaled)

print("\nOriginal :", scaled.shape)
print("\nReduced:", pca_data.shape)

pca_df= pd.DataFrame(data=pca_data, columns=['PC1','PC2'])
pca_df['quality']= y.values



plt.figure(figsize=(10,8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='quality', palette='Set1')
plt.title("PCA of Wine Quality Dataset", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()