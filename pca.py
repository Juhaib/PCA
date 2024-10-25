# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Task 1: Exploratory Data Analysis (EDA)
data = pd.read_csv('wine.csv')

# Check the column names
print("Column names:", data.columns.tolist())  # Print all column names

print(data.info())
print(data.describe())
print(data.head())

# Plot histograms for each feature
data.hist(figsize=(15, 10))
plt.show()

# Replace 'actual_feature_name' with a valid feature name from your dataset
valid_feature_name = data.columns[0]  # Example: using the first column name
sns.boxplot(data[valid_feature_name])  
plt.title(f'Boxplot for {valid_feature_name}')
plt.show()

# Investigate correlations between features
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Task 2: Dimensionality Reduction with PCA
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

pca = PCA()
pca.fit(standardized_data)

# Scree Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Cumulative Explained Variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.show()

optimal_components = 2  # Adjust based on the previous analysis
pca = PCA(n_components=optimal_components)
pca_data = pca.fit_transform(standardized_data)

# Task 3: Clustering with Original Data
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(standardized_data)
data['Cluster'] = clusters

# Visualize clustering results
# Replace 'actual_feature1' and 'actual_feature2' with valid feature names
feature1 = data.columns[1]  # Example: using the second column
feature2 = data.columns[2]  # Example: using the third column
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[feature1], y=data[feature2], hue=data['Cluster'], palette='viridis')
plt.title('K-means Clustering on Original Data')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.show()

silhouette = silhouette_score(standardized_data, clusters)
db_index = davies_bouldin_score(standardized_data, clusters)
print(f'Silhouette Score: {silhouette}')
print(f'Davies-Bouldin Index: {db_index}')

# Task 4: Clustering with PCA Data
kmeans_pca = KMeans(n_clusters=3)
clusters_pca = kmeans_pca.fit_predict(pca_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters_pca, cmap='viridis')
plt.title('K-means Clustering on PCA-Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Task 5: Comparison and Analysis
# (Add any additional analysis or visualizations here)

# Task 6: Conclusion and Insights
# (Summarize findings and insights)
