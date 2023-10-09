import numpy as np
from pathlib import Path 
from sklearn.cluster import DBSCAN

import config as cf
from ants import main, plot_path

NAME = "main"
NUM_STEPS = 3000
INIT_ANTS = 70
MAX_ANTS = 70

# Path("my_imgs_2").mkdir(parents=True, exist_ok=True)
Path("my_imgs_dbscan_2").mkdir(parents=True, exist_ok=True)

# standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":

    num_round_trips, paths, coeff, distances = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
        ant_only_gif=False,
    )

    print(f"num_round_trips {num_round_trips} / coeff {coeff / MAX_ANTS}")
    # f = open(f"my_imgs_2/{NAME}.txt", "w")
    # f = open(f"my_imgs_dbscan/{NAME}.txt", "w")
    f = open(f"my_imgs_dbscan_2/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff / MAX_ANTS}")
    f.close()

    
    for i in range(len(paths)):
        # plot_path(np.random.choice(paths), f"my_imgs_2/path_{i}.png")
        # plot_path(np.random.choice(paths), f"my_imgs_dbscan/path_{i}.png")
        plot_path(np.random.choice(paths), f"my_imgs_dbscan_2/path_{i}.png")

    plt.plot(distances, NUM_STEPS)

    data_points = [point for trajectory in paths for point in trajectory]

    # Convert data_points to a NumPy array
    data_points_array = np.array(data_points)

    # Apply DBSCAN with appropriate parameters
    eps = 0.1  # Adjust based on the characteristic distance between pheromone deposits
    min_samples = 5  # Adjust based on the minimum number of deposits required to form a cluster

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data_points_array)

    # Now 'cluster_labels' contains cluster assignments for each data point
    # -1 indicates noise points, and other values represent cluster labels

    # Analyze the clusters and measure stigmergy
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters) - 1  # Exclude noise cluster (-1)

    # Measure cluster characteristics (e.g., size, density, centrality)
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_clusters if label != -1]

    # Calculate stigmergy-related metrics based on cluster analysis
    average_cluster_size = np.mean(cluster_sizes)
    cluster_density = len(data_points) / num_clusters  # Total deposits divided by the number of clusters

    # You can further analyze and visualize cluster properties to assess stigmergy
    print(f"\ncluster_labels: {cluster_labels}")
    print(f"unique_clusters: {unique_clusters}")
    print(f"num_clusters: {num_clusters}")
    print(f"cluster_sizes: {cluster_sizes}")
    print(f"average_cluster_size: {average_cluster_size}")
    print(f"cluster_density: {cluster_density}\n")
