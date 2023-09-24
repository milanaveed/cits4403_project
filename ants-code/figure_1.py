import numpy as np
from pathlib import Path 

import config as cf
from ants import main, plot_path

NAME = "main"  # Name for this simulation
NUM_STEPS = 2000  # Number of simulation steps
INIT_ANTS = 70  # Initial number of ants
MAX_ANTS = 70  # Maximum number of ants in the simulation

Path("imgs").mkdir(parents=True, exist_ok=True)  # Create a directory for saving images if doesn't exist

# Define a standard prior for observations
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):  # Assign values to the observation prior
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":
    # Run the main simulation
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
        ant_only_gif=False,
    )

    # Print simulation results
    print(f"num_round_trips {num_round_trips} / coeff {coeff / MAX_ANTS}")

    # Save simulation results to a text file
    f = open(f"imgs/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff / MAX_ANTS}")
    f.close()

    # Generate and save path images for visualization
    for i in range(len(paths)):
        plot_path(np.random.choice(paths), f"imgs/path_{i}.png")
