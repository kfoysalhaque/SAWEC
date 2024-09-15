import numpy as np
from sklearn.cluster import DBSCAN

def dbscan(loc_file, eps=3, min_samples=10):
    # Assuming you have loaded the combined data
    clusters = np.load(loc_file)

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(clusters)
    return clusters, cluster_labels
