import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read College_Data dataset from the repository and print some of its information
df = pd.read_csv('College_Data',index_col=0)
print(df.head())
print(df.info())
print(df.describe())

# Scatter Plot between two variables so it's possible to see how they correlate
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='inferno',size=6,aspect=1,fit_reg=False)
plt.show()

# Histogram Plot between the results of the column 'Outstate' vs 'Private'
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='magma',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
plt.show()

# print(df[df['Grad.Rate'] > 200])
# print(df['Grad.Rate']['Cazenovia College'] = 200)

# Import K-Means-Clustering from sklearn, declare kmeans with 2 clusters and fit df without column 'Private'
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))

# Print Cluster centers to better understand where they're located
print(kmeans.cluster_centers_)

# Function that returns 1 if cluster == 'Yes' and 0 if cluster == "No"
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

# Apply converter function in column 'Private'
df['Cluster'] = df['Private'].apply(converter)
print(df.head())

# Import some metrics and print the matrixes to understand how the model is behaving
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))