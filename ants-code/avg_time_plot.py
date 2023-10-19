import numpy as np
from pathlib import Path 
# from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import config as cf
from ants import main, plot_path

NAME = "main"
NUM_STEPS = 1000
# INIT_ANTS = 70
# MAX_ANTS = 70
ANT_POPS = [10, 30, 50, 70]
ANT_POP_DISTS = []

# Path("my_imgs_2").mkdir(parents=True, exist_ok=True)
# Path("my_imgs_dbscan_2").mkdir(parents=True, exist_ok=True)
Path("time_imgs").mkdir(parents=True, exist_ok=True)

# standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":

    for i in range(len(ANT_POPS)):

        paths, coeff, distances, avg_round_trip_time = main(
            num_steps=NUM_STEPS,
            init_ants=ANT_POPS[i],
            max_ants=ANT_POPS[i],
            C=C,
            save=True,
            switch=True,
            name=NAME,
            ant_only_gif=False,
        )

        # ANT_POP_DISTS.append(distances)

        print(f"ANT_POPS[i]: {ANT_POPS[i]}")
        print(f"avg_round_trip_time: {avg_round_trip_time}")
        # print(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
        # f = open(f"my_imgs_2/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan/{NAME}.txt", "w")
        # f = open(f"my_imgs_dbscan_2/{NAME}.txt", "w")

        # f = open(f"my_imgs_dbscan_70_ants/{NAME}.txt", "w")
        # f.write(f"num_round_trips {num_round_trips} / coeff {coeff / ANT_POPS[i]}")
        # f.close()

    plt.title("Average Round Trip Time Over Time")
    plt.xlabel("Time")
    plt.ylabel("AVG Round Trip Time")
    # plt.plot([step for step in range(NUM_STEPS)], ANT_POP_DISTS[0])
    # plt.plot([step for step in range(NUM_STEPS)], ANT_POP_DISTS[1])
    # plt.plot([step for step in range(NUM_STEPS)], ANT_POP_DISTS[2])
    # plt.plot([step for step in range(NUM_STEPS)], ANT_POP_DISTS[3])
    # plt.show()

    # Plot the data with labels
    for i in range(4):
        plt.plot(
            [range(NUM_STEPS)], 
            avg_round_trip_time, 
            label=f'{ANT_POPS[i]}'
        )

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()