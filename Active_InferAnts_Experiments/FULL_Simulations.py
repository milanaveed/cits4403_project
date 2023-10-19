import numpy as np
from pathlib import Path 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import config as cf
from ants import main, plot_path

NAME = "main"
NUM_STEPS = 500
# INIT_ANTS = 70
# MAX_ANTS = 70
ANT_POPS = [10, 30, 50, 70, 110]
ANT_POPS_COLOURS = ["blue", "orange", "limegreen", "red", "purple"]
ANT_POP_DISTS = []
AVG_NUM_RT_PER_POP_SIZE_PER_TIME = []
MORAN_IS = []
CLUSTER_DENSITIES = []
NUMBER_CLUSTERS = []

NUM_SIMS = len(ANT_POPS)

run_ctr = 0

# Path("my_imgs_2").mkdir(parents=True, exist_ok=True)
# Path("my_imgs_dbscan_2").mkdir(parents=True, exist_ok=True)
# Path("my_imgs_dbscan_70_ants").mkdir(parents=True, exist_ok=True)

# standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":

    for i in range(NUM_SIMS):

        num_round_trips, paths, coeff, distances, num_round_trips_per_time, ants, cluster_densities, num_clusters, Morans_i_values = main(
            num_steps=NUM_STEPS,
            init_ants=ANT_POPS[i],
            max_ants=ANT_POPS[i],
            C=C,
            save=True,
            switch=True,
            name=NAME,
            ant_only_gif=False,
            num_runs = NUM_SIMS,
            ctr = run_ctr
        )

        ANT_POP_DISTS.append(distances)
        CLUSTER_DENSITIES.append(cluster_densities)
        NUMBER_CLUSTERS.append(num_clusters)
        AVG_NUM_RT_PER_POP_SIZE_PER_TIME.append(num_round_trips_per_time)
        MORAN_IS.append(Morans_i_values)

        # print(f"ANT_POPS[i]: {ANT_POPS[i]}")
        # print(f"num_round_trips_per_time: {num_round_trips_per_time}")
        # print(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")

        # f = open(f"my_imgs_2/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan_2/{NAME}.txt", "w")

        f = open(f"my_imgs_dbscan_70_ants/{NAME}.txt", "w")
        f.write(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
        f.close()

        run_ctr += 1

    plt.title("Average Ant Distance Over Time")
    plt.xlabel("Time")
    plt.ylabel("AVG Ant Dist")
    for i in range(len(ANT_POPS)):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            ANT_POP_DISTS[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

    plt.title("Average Number of RTs Over Time")
    plt.xlabel("Time")
    plt.ylabel("AVG Num RTs")
    for i in range(len(ANT_POPS)):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            AVG_NUM_RT_PER_POP_SIZE_PER_TIME[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

    plt.title("Average DBSCAN Cluster Densities")
    plt.xlabel("Time")
    plt.ylabel("AVG Cluster Densities")
    for i in range(len(ANT_POPS)):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            CLUSTER_DENSITIES[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

    plt.title("Number of DBSCAN Clusters")
    plt.xlabel("Time")
    plt.ylabel("Num Clusters")
    for i in range(len(ANT_POPS)):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            NUMBER_CLUSTERS[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

    plt.title("Moran's I Between Ants and Pheromones")
    plt.xlabel("Time")
    plt.ylabel("Moran's I")
    for i in range(len(ANT_POPS)):
        plt.plot(
            [step for step in range(NUM_STEPS)], 
            MORAN_IS[i], 
            label=f'{ANT_POPS[i]}',
            color = ANT_POPS_COLOURS[i]
        )
    plt.legend()
    plt.show()

