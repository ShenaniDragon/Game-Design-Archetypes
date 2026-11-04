import kagglehub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA

# Getting the data
path = kagglehub.dataset_download("omkarbhujbalrao/pokemon-dataset-gen-1-9")

dataset_path = path
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
df_base = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
print(df_base)

# Cleaning Data
stats = ['HP', 'Attack', 'Defense', 'Sp.Atk', 'Sp.Def', 'Speed']
selection = df_base.loc[:,'Is_Legendary':'Is_Pseudo_Legendary'].columns.tolist()
to_drop = ['Height(m)','Weight{kg}'] + selection
df_base = df_base.drop(to_drop, axis =1)
columns_to_dropna = [c for c in df_base.columns if c not in 'Type_2']
print(columns_to_dropna)
df = df_base.dropna(subset = columns_to_dropna)
df = df[df['Total_Stats'] > 450]
print(df)

# K Mean
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['HP','Attack','Defense','Sp.Atk','Sp.Def','Speed']])

K = range(1, 11)

inertias = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)    
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Using Found K
k = 5

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# cluster names (came back from later code to pinpoint the actual names)
cluster_name = {
    0 : 'Atk Sweeper',
    1 : 'Sp.Atk Juggernaut',
    2 : 'Jack of all Trade',
    3 : 'Resistance Tank',
    4 : 'Atk Juggernaut',
}

df['Cluster_Label'] = df['Cluster'].map(cluster_name)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Graphing PCA
for label in df['Cluster_Label'].unique():
    subset = df[df['Cluster_Label'] == label]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=label, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Clusters in PCA space')
plt.legend(title= 'Archetypes', prop={'size': 8})
plt.show()

# Graphing common Typings to Archetype
clusters = df['Cluster_Label'].unique()
n_clusters = len(clusters)

fig, axes = plt.subplots(3, 2, figsize=(6, 8))
axes = axes.flatten()

for i, cluster in enumerate(clusters):
    subset = df[df['Cluster_Label'] == cluster]
    types_combined = pd.concat([subset['Type_1'], subset['Type_2'].dropna()])
    type_counts = types_combined.value_counts()
    top5 = type_counts.head(5)
    other_count = type_counts[5:].sum()
    if other_count > 0:
        top5['Other'] = other_count

    axes[i].pie(top5, labels=top5.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    axes[i].set_title(f'Archetype: {cluster}')


if n_clusters < len(axes):
    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')

plt.tight_layout()
plt.show()

# Graphing stat Averages
cluster_stats = df.groupby('Cluster_Label')[stats].mean()

plt.figure(figsize=(8,6))
sns.heatmap(cluster_stats, annot=True,fmt='.0f', cmap='YlGnBu')
plt.title('Average Stats per Cluster')
plt.show()
