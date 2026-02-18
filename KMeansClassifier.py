def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(filepath):

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} cars from {filepath}")
    return df

def run_kmeans(features, n_clusters=5, random_state=42):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(features)
    print(f"K-Means complete. Cluster centers:\n{kmeans.cluster_centers_}")
    return clusters, kmeans

def assign_cluster_styles(df, clusters):

    df["Cluster"] = clusters
    cluster_style_map = {}

    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_rows = df[df["Cluster"] == cluster_id]
        majority_style = cluster_rows["Style"].value_counts().idxmax()
        cluster_style_map[cluster_id] = majority_style
        print(f"Cluster {cluster_id} -> {majority_style}")

    df["ClusterStyle"] = df["Cluster"].map(cluster_style_map)
    return df, cluster_style_map

def save_cluster_cars(df, output_path):

    output = df[["Volume", "Doors", "Style", "ClusterStyle"]]
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def save_cluster_accuracy(df, cluster_style_map, output_path):

    rows = []
    for cluster_id, cluster_style in cluster_style_map.items():
        cluster_rows = df[df["Cluster"] == cluster_id]
        size = len(cluster_rows)
        correct = (cluster_rows["Style"] == cluster_style).sum()
        accuracy = round(correct / size, 4)
        rows.append({
            "ClusterStyle": cluster_style,
            "SizeOfCluster": size,
            "Accuracy": accuracy
        })
        print(f"ClusterStyle: {cluster_style}, Size: {size}, Accuracy: {accuracy:.4f}")

    accuracy_df = pd.DataFrame(rows)
    accuracy_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
#--------------------------------------------------------------------------------------------------
def main():

    df = load_data("AllCars.csv")

    features = df[["Volume", "Doors"]].to_numpy()
    clusters, kmeans = run_kmeans(features, n_clusters=5, random_state=42)
    df, cluster_style_map = assign_cluster_styles(df, clusters)

#----Save 
    save_cluster_cars(df, "ClusterCars.csv")
    save_cluster_accuracy(df, cluster_style_map, "ClusterAccuracy.csv")
#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------
