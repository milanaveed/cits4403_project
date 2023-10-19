import numpy as np
from pathlib import Path 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import config as cf
from ants import main, plot_path

NAME = "main"
NUM_STEPS = 1000
# INIT_ANTS = 70
# MAX_ANTS = 70
ANT_POPS = [10, 30, 50, 70]
ANT_POPS_COLOURS = ["blue", "orange", "limegreen", "red"]
ANT_POP_DISTS = []
AVG_NUM_RT_PER_POP_SIZE_PER_TIME = []

CLUSTER_DENSITIES = []
MORANS_I_VALUES = []


# Path("my_imgs_2").mkdir(parents=True, exist_ok=True)
# Path("my_imgs_dbscan_2").mkdir(parents=True, exist_ok=True)
Path("my_imgs_dbscan_70_ants").mkdir(parents=True, exist_ok=True)

# standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":

    for i in range(len(ANT_POPS)):

        num_round_trips, paths, coeff, distances, num_round_trips_per_time, ants, cluster_densities, Morans_i_values = main(
            num_steps=NUM_STEPS,
            init_ants=ANT_POPS[i],
            max_ants=ANT_POPS[i],
            C=C,
            save=True,
            switch=True,
            name=NAME,
            ant_only_gif=False,
        )

        ANT_POP_DISTS.append(distances)
        CLUSTER_DENSITIES.append(cluster_densities)
        MORANS_I_VALUES.append(Morans_i_values)

        AVG_NUM_RT_PER_POP_SIZE_PER_TIME.append(num_round_trips_per_time)

        # print(f"ANT_POPS[i]: {ANT_POPS[i]}")
        # print(f"num_round_trips_per_time: {num_round_trips_per_time}")
        # print(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
        # f = open(f"my_imgs_2/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan_2/{NAME}.txt", "w")

        f = open(f"my_imgs_dbscan_70_ants/{NAME}.txt", "w")
        f.write(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
        f.close()

        # for i in range(len(paths)):
        #     # plot_path(np.random.choice(paths), f"my_imgs_2/path_{i}.png")
        #     # plot_path(np.random.choice(paths), f"my_imgs_dbscan/path_{i}.png")
        #     # plot_path(np.random.choice(paths), f"my_imgs_dbscan_2/path_{i}.png")
        #     plot_path(np.random.choice(paths), f"my_imgs_dbscan_70_ants/path_{i}.png")

        # # DBSCAN Clustering
        # data_points = [point for trajectory in paths for point in trajectory]

        # # Convert data_points to a NumPy array
        # data_points_array = np.array(data_points)

        # # Apply DBSCAN with appropriate parameters
        # eps = 0.1  # Adjust based on the characteristic distance between pheromone deposits
        # min_samples = 5  # Adjust based on the minimum number of deposits required to form a cluster

        # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # cluster_labels = dbscan.fit_predict(data_points_array)

        # # Now 'cluster_labels' contains cluster assignments for each data point
        # # -1 indicates noise points, and other values represent cluster labels

        # # Analyze the clusters and measure stigmergy
        # unique_clusters = np.unique(cluster_labels)
        # num_clusters = len(unique_clusters) - 1  # Exclude noise cluster (-1)

        # # Measure cluster characteristics (e.g., size, density, centrality)
        # cluster_sizes = [np.sum(cluster_labels == label) for label in unique_clusters if label != -1]

        # # Calculate stigmergy-related metrics based on cluster analysis
        # average_cluster_size = np.mean(cluster_sizes)
        # cluster_density = len(data_points) / num_clusters  # Total deposits divided by the number of clusters

        # CLUSTER_DENSITIES.append(cluster_density)

        # # You can further analyze and visualize cluster properties to assess stigmergy
        # print(f"\ncluster_labels: {cluster_labels}")
        # print(f"unique_clusters: {unique_clusters}")
        # print(f"num_clusters: {num_clusters}")
        # print(f"cluster_sizes: {cluster_sizes}")
        # print(f"average_cluster_size: {average_cluster_size}")
        # print(f"cluster_density: {cluster_density}\n")

    plt.title("Average DBSCAN Cluster Densities")
    plt.xlabel("Time")
    plt.ylabel("AVG Cluster Densities")
    for i in range(4):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            CLUSTER_DENSITIES[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

    plt.title("Morans I Values")
    plt.xlabel("Time")
    plt.ylabel("AVG Cluster Densities")
    for i in range(4):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            MORANS_I_VALUES[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()