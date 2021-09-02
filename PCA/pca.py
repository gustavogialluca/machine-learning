import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Load Breast Cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

# Save dataset as Pandas datafra,e
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())

# Import scaler, fit the data into scaler and transform it
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# Import PCA, fit data into PCA and transform it to 2 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print("Scaled Data Shape: " + str(scaled_data.shape))
print("X_PCA Data Shape: " + str(x_pca.shape))

# Plot those 2 dimensions got from the PCA
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# Plot how those 2 dimensions are compared to the original data from the cancer dataset
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)