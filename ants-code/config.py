import numpy as np

# Below are configuration settings to define parameters and control various aspects of the simulation

ADD_ANT_EVERY = 50  # An integer representing the frequency at which ants are added to the simulation
INIT_X = 20  # Initial coordinate x for the ants when they start in the simulation
INIT_Y = 30  # Initial coordinate y for the ants when they start in the simulation

NEST_FACTOR = 0.1  # ? A floating-point number that likely represents a factor related to the nest

GRID = [40, 40]  # Define the grid size
GRID_SIZE = np.prod(GRID)  # Calculate the total number of grid cells

FOCAL_AREA = [3, 3]  # Dimensions of a focal area within the grid
FOCAL_SIZE = np.prod(FOCAL_AREA)  # Calculate the total number of cells within the focal area
ACTION_MAP = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]  # Define possible directions of one move
OPPOSITE_ACTIONS = list(reversed(range(len(ACTION_MAP))))  # Reversed order of possible directions of one move

FOOD_LOCATION = [40, 5]  # Initial location of the food source
FOOD_SIZE = [10, 10]  # The size of the food resource area

# ? These variables likely define the positions of walls or barriers in the simulation
WALL_LEFT = 15
WALL_RIGHT = 25
WALL_TOP = 10

NUM_PHEROMONE_LEVELS = 10  # The number of discrete levels for pheromone intensity
DECAY_FACTOR = 0.01  # Define the rate at which pheromone intensity decays over time

NUM_OBSERVATIONS = NUM_PHEROMONE_LEVELS  # ? The number of possible observations, likely related to pheromone levels
NUM_STATES = FOCAL_SIZE  # ? The number of possible states within the focal area
NUM_ACTIONS = FOCAL_SIZE  # ? The number of possible actions within the focal area

MAX_LEN = 500 # ?  A maximum length or threshold value, possibly related to the simulation's duration or iteration limit
