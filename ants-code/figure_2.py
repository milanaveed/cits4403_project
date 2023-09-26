import numpy as np
from pathlib import Path 

import config as cf
from ants import main

Path("imgs").mkdir(parents=True, exist_ok=True)


NUM_AVERAGE = 5  # Define the number of simulations to average
NAME = "compare"  # Name for this simulation
NUM_STEPS = 2000  # Number of simulation steps
INIT_ANTS = 200  # Initial number of ants
MAX_ANTS = 200  # Maximum number of ants in the simulation

# Define a standard prior for observations
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):  # Assign values to the observation prior
    C[o] = prior
    C[o] = prior
    prior += PRIOR_TICK


def run():
    num_round_trips, paths, coeff = main(
        # Run the simulation
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
    )

    # Print simulation results
    print(f"num_round_trips_{MAX_ANTS} {num_round_trips} / coeff {coeff}")

    # Save simulation results to a text file
    f = open(f"imgs/{NAME}.txt", "a+")
    f.write(f"num_round_trips_{MAX_ANTS} {num_round_trips} / coeff {coeff}\n")
    f.close()


if __name__ == "__main__":
    # Run multiple simulations with different parameters and average the results
    for _ in range(NUM_AVERAGE):
        # Simulation with 10 initial ants
        NAME = "compare_10"
        INIT_ANTS = 10
        MAX_ANTS = 10
        run()

        # Simulation with 30 initial ants
        NAME = "compare_30"
        INIT_ANTS = 30
        MAX_ANTS = 30
        run()

        # Simulation with 50 initial ants
        NAME = "compare_50"
        INIT_ANTS = 50
        MAX_ANTS = 50
        run()
        
        # Simulation with 70 initial ants
        NAME = "compare_70"
        INIT_ANTS = 70
        MAX_ANTS = 70
        run()
