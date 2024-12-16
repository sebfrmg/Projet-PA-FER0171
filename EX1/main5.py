import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

def load_mnist_data(train_path, test_path, sample_size=5000):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    data = pd.concat([train_data, test_data], axis=0)
    data = data.sample(n=sample_size, random_state=42)
    
    y_labels = data.iloc[:, 0].values  
    x_data = data.iloc[:, 1:].values  
    print(f"Donnees chargees : {x_data.shape[0]} echantillons, {x_data.shape[1]} dimensions")
    return x_data, y_labels

def reduce_dimensions(data, n_components=20):
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    print(f"Dimensionnalite reduite a {n_components} dimensions.")
    return data_reduced

def compute_affinity_propagation(data_chunk):
    print("Execution de Affinity Propagation sur un sous-ensemble...")
    ap = AffinityPropagation(damping=0.9, max_iter=200, random_state=42)
    ap.fit(data_chunk)
    return ap.labels_, ap.cluster_centers_indices_

def parallel_affinity_propagation(data, n_jobs=-1):
    n_chunks = 4  
    data_chunks = np.array_split(data, n_chunks)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_affinity_propagation)(chunk) for chunk in data_chunks
    )
    labels = np.concatenate([result[0] for result in results])
    centers = np.concatenate([result[1] for result in results if result[1] is not None])
    print(f"Clusters combines a partir des sous-ensembles : {len(np.unique(labels))} clusters trouves.")
    return labels, centers

def visualize_clusters(data, labels, sample_size=100):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:sample_size, 0], data[:sample_size, 1], c=labels[:sample_size], cmap='viridis', s=10)
    plt.title("Clustering Affinity Propagation (Parallele, Echantillon reduit)")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.colorbar(label="Cluster")
    plt.show()

def main():
    train_path = "EX1/mnist_train.csv"
    test_path = "EX1/mnist_test.csv"
    x_data, y_labels = load_mnist_data(train_path, test_path, sample_size=5000)
    x_reduced = reduce_dimensions(x_data, n_components=20)
    labels, centers = parallel_affinity_propagation(x_reduced)
    x_pca_2d = reduce_dimensions(x_reduced, n_components=2)
    visualize_clusters(x_pca_2d, labels)

if __name__ == "__main__":
    main()
