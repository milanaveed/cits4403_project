import numpy as np
from pathlib import Path 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import config as cf
from ants import main, plot_path

NAME = "main"
NUM_STEPS = 1500
# INIT_ANTS = 70
# MAX_ANTS = 70
ANT_POPS = [10, 30, 50, 70]
ANT_POPS_COLOURS = ["blue", "orange", "limegreen", "red"]
ANT_POP_DISTS = []
AVG_NUM_RT_PER_POP_SIZE_PER_TIME = []

CLUSTER_DENSITIES = []
# MORANS_I_VALUES = []


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

    # for i in range(len(ANT_POPS)):

    num_round_trips, paths, coeff, distances, num_round_trips_per_time, ants, cluster_densities, Morans_i_values = main(
        num_steps=NUM_STEPS,
        init_ants=ANT_POPS[3],
        max_ants=ANT_POPS[3],
        C=C,
        save=True,
        switch=True,
        name=NAME,
        ant_only_gif=False,
    )

    # ANT_POP_DISTS.append(distances)
    # CLUSTER_DENSITIES.append(cluster_densities)
    # MORANS_I_VALUES.append(Morans_i_values)
    # AVG_NUM_RT_PER_POP_SIZE_PER_TIME.append(num_round_trips_per_time)

    # print(f"ANT_POPS[i]: {ANT_POPS[i]}")
    # print(f"num_round_trips_per_time: {num_round_trips_per_time}")
    # print(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
    # f = open(f"my_imgs_2/{NAME}.txt", "w")
    # f = open(f"my_imgs_dbscan/{NAME}.txt", "w")
    # f = open(f"my_imgs_dbscan_2/{NAME}.txt", "w")

    f = open(f"my_imgs_dbscan_70_ants/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[3]}")
    f.close()

    # plt.title("Average DBSCAN Cluster Densities")
    # plt.xlabel("Time")
    # plt.ylabel("AVG Cluster Densities")
    # # for i in range(4):
    # plt.plot(
    #     [step for step in range(NUM_STEPS)], 
    #     cluster_densities, 
    #     label=f'{ANT_POPS[3]}',
    #     color = ANT_POPS_COLOURS[3]
    # )
    # plt.legend()
    # plt.show()

    plt.title("Moran's I Between Ants and Pheromones")
    plt.xlabel("Time")
    plt.ylabel("Moran's I")
    plt.plot(
        [step for step in range(NUM_STEPS)], 
        Morans_i_values, 
        label=f'{ANT_POPS[3]}',
        color = ANT_POPS_COLOURS[3]
    )
    plt.legend()
    plt.show()