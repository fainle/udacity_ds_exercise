import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def simulate_data(n=500, features=10, centroids=3):
    dataset, y = make_blobs(n_samples=n, n_features=features, centers=centroids, random_state=42)
    
    return dataset


def plot_data(data, labels):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10')
    

data = simulate_data(200, 5, 4)


def get_kmeans_score(data, center):
    kmeans = KMeans(n_clusters=center)
    
    model = kmeans.fit(data)
    score = np.ads(model.score(data))
    
    return score


def fit_mods():
    scores = []
    centers = list(range(1, 11))
    
    for center in centers:
        scores.append(get_kmeans_score(data, center))
        
    return centers, scores