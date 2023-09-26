import config as cf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio

matplotlib.use("Agg")

class Ant(object):
    '''
    An Ant class represents each ant agent and its behavior.
    '''
    def __init__(self, mdp, init_x, init_y):
        '''
        Constructor method that initializes an instance of the `Ant` class.
        '''
        self.mdp = mdp  # ?  Store the Markov Decision Process (MDP) related to the ant's behavior
        self.x_pos = init_x  # Initial x coordinate for the ant position
        self.y_pos = init_y  # Initial y coordinate for the ant position
        self.traj = [(init_x, init_y)]  # A list to store the trajectory of the ant
        self.distance = []  # A list to store the distances between the ant's current position and the initial position
        self.backward_step = 0  # Represnet the number of backward steps taken by the ant
        self.is_returning = False  # Whether the ant is currently returning to the nest

    def update_forward(self, x_pos, y_pos):
        '''
        Update the ant's position, trajectory and distance when it moves forward.
        '''
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.traj.append((x_pos, y_pos))
        self.distance.append(dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))

    def update_backward(self, x_pos, y_pos):
        '''
        Update the ant's position and distance when it moves backward.
        '''
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.distance.append(dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))


class Env(object):
    '''
    An Env class represents the environment within which the ants move and interact.
    This class contains methods and attributes to manage the environment, including updating ant positions, handling observations, checking for obstacles, and visualizing the environment. 
    '''
    def __init__(self):
        '''
        Constructor method that initializes an instance of the `Env` class.
        '''
        self.visit_matrix = np.zeros((cf.GRID[0], cf.GRID[1]))  # A 2D array to track the visitation count of each location 
        self.obs_matrix = np.zeros((cf.NUM_OBSERVATIONS, cf.GRID[0], cf.GRID[1])) # A 3D matrix to represent the observation matrix
        self.obs_matrix[0, :, :] = 1.0  # Initialize the observation matrix with all ones in the first observation, indicating that all locations are initially observable

    def get_A(self, ant):
        '''
        Return the observation matrix A for a given ant's position.
        '''
        A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
        for s in range(cf.NUM_STATES):
            delta = cf.ACTION_MAP[s]
            A[:, s] = self.obs_matrix[:, ant.x_pos + delta[0], ant.y_pos + delta[1]]
        return A

    def get_obs(self, ant):
        '''
        Return the observation index for a given ant's position.
        '''
        obs_vec = self.obs_matrix[:, ant.x_pos, ant.y_pos]
        return np.argmax(obs_vec)

    def check_food(self, x_pos, y_pos):
        '''
        Check if a given position contains food.
        '''
        is_food = False
        if (x_pos > (cf.FOOD_LOCATION[0] - cf.FOOD_SIZE[0])) and (
            x_pos < (cf.FOOD_LOCATION[0] + cf.FOOD_SIZE[0])
        ):
            if (y_pos > (cf.FOOD_LOCATION[1] - cf.FOOD_SIZE[1])) and (
                y_pos < (cf.FOOD_LOCATION[1] + cf.FOOD_SIZE[1])
            ):
                is_food = True
        return is_food

    def check_walls(self, orig_x, orig_y, x_pos, y_pos):
        '''
        Check if an ant encounters walls or obstacles.
        '''
        valid = True
        if orig_y > cf.WALL_TOP:
            if orig_x >= cf.WALL_LEFT and x_pos <= cf.WALL_LEFT:
                valid = False
            if orig_x <= cf.WALL_RIGHT and x_pos >= cf.WALL_RIGHT:
                valid = False
        if orig_y <= cf.WALL_TOP:
            if y_pos > cf.WALL_TOP and ((x_pos < cf.WALL_LEFT) or (x_pos > cf.WALL_RIGHT)):
                valid = False
        return valid

    def step_forward(self, ant, action):
        '''
        Update an ant's position based on the selected action (direction).
        '''
        delta = cf.ACTION_MAP[action]
        x_pos = np.clip(ant.x_pos + delta[0], 1, cf.GRID[0] - 2)
        y_pos = np.clip(ant.y_pos + delta[1], 1, cf.GRID[1] - 2)

        if self.check_food(x_pos, y_pos) and np.random.rand() < cf.NEST_FACTOR:
            # If an ant finds food, it starts returning to the nest
            ant.is_returning = True
            ant.backward_step = 0

        if self.check_walls(ant.x_pos, ant.y_pos, x_pos, y_pos):
            # Check for wall collisions and update the ant's position accordingly
            ant.update_forward(x_pos, y_pos)

        # Code for limiting the length of an ant's trajectory
        """
        if len(ant.traj) > cf.MAX_LEN:
            pos = ant.traj[0]
            ant.update_backward(pos[0], pos[1])
            ant.traj = []
        """

    def step_backward(self, ant):
        '''
        Backward step for an ant returning to the nest.
        '''
        path_len = len(ant.traj)
        next_step = path_len - (ant.backward_step + 1)
        pos = ant.traj[next_step]
        ant.update_backward(pos[0], pos[1])

        # Update visitation count and observation matrix
        self.visit_matrix[pos[0], pos[1]] += 1
        curr_obs = np.argmax(self.obs_matrix[:, pos[0], pos[1]])
        curr_obs = min(curr_obs + 1, cf.NUM_OBSERVATIONS - 1)

        self.obs_matrix[:, pos[0], pos[1]] = 0.0
        self.obs_matrix[curr_obs, pos[0], pos[1]] = 1.0

        ant.backward_step += 1
        if ant.backward_step >= path_len - 1:
            ant.is_returning = False
            traj = ant.traj
            ant.traj = []
            return True, traj
        else:
            return False, None

    def decay(self):
        ''' 
        Decay the observation matrix over time.
        '''
        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                if (curr_obs > 0) and (np.random.rand() < cf.DECAY_FACTOR):
                    curr_obs = curr_obs - 1
                    self.obs_matrix[:, x, y] = 0.0
                    self.obs_matrix[curr_obs, x, y] = 1.0

    def plot(self, ants, savefig=False, name="", ant_only_gif=False):
        '''
        Plot the environment, ant trajectories, and sensory cues.
        '''
        x_pos_forward, y_pos_forward = [], []
        x_pos_backward, y_pos_backward = [], []
        for ant in ants:
            if ant.is_returning:
                x_pos_backward.append(ant.x_pos)
                y_pos_backward.append(ant.y_pos)
            else:
                x_pos_forward.append(ant.x_pos)
                y_pos_forward.append(ant.y_pos)

        img = np.ones((cf.GRID[0], cf.GRID[1]))
        fig, ax = plt.subplots()
        ax.imshow(img.T, cmap="gray")
        plot_matrix = np.zeros((cf.GRID[0], cf.GRID[1]))

        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                plot_matrix[x, y] = curr_obs

        if ant_only_gif == False:
            ax.imshow(plot_matrix.T, alpha=0.7, vmin=0)

        if not savefig:
            ax.scatter(x_pos_forward, y_pos_forward, color="red", s=5)
            ax.scatter(x_pos_backward, y_pos_backward, color="blue", s=5)

        if not savefig:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close("all")
            return img
        else:
            plt.savefig(name)
        plt.close("all")


class MDP(object):
    '''
    The MDP class represents a Markov Decision Process (MDP) used in the ant simulation.
    It initializes the MDP with transition matrices A, B, and C, and provides methods for state inference, action inference, and other MDP-related operations.
    '''
    def __init__(self, A, B, C):
        '''
        Constructor method that initializes an instance of the `MDP` class.

        Args:
            A (numpy.ndarray): Transition matrix A.
            B (numpy.ndarray): Transition matrix B.
            C (numpy.ndarray): Transition matrix C.
        '''
        self.A = A
        self.B = B
        self.C = C
        self.p0 = np.exp(-16)

        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.num_actions = self.B.shape[0]

        # Modify A, B, and C matrices with p0 and normalize
        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.B = self.B + self.p0
        for a in range(self.num_actions):
            self.B[a] = self.normdist(self.B[a])

        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.num_states, 1])
        self.uQ = np.zeros([self.num_actions, 1])
        self.prev_action = None
        self.t = 0

    def set_A(self, A):
        '''
        Set the transition matrix A.

        Args:
            A (numpy.ndarray): New transition matrix A.
        '''
        self.A = A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

    def reset(self, obs):
        '''
        Reset the MDP with a new observation.

        Args:
            obs (int): The new observation.
        '''
        self.t = 0
        self.curr_obs = obs
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.prev_action = self.random_action()

    def step(self, obs):
        '''
        Perform a step in the MDP.

        Args:
            obs (int): The current observation.

        Returns:
            int: The selected action.
        '''
        # State inference
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.prev_action], self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(prior)

        # Action inference
        SCALE = 10
        neg_efe = np.zeros([self.num_actions, 1])
        for a in range(self.num_actions):
            fs = np.dot(self.B[a], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)
            utility = np.sum(fo * np.log(fo / self.C), axis=0)
            utility = utility[0]
            neg_efe[a] -= utility / SCALE

        # Priors
        neg_efe[4] -= 20.0
        neg_efe[cf.OPPOSITE_ACTIONS[self.prev_action]] -= 20.0  # type: ignore

        # Action selection
        self.uQ = self.softmax(neg_efe)
        action = np.argmax(np.random.multinomial(1, self.uQ.squeeze()))
        self.prev_action = action
        return action

    def random_action(self):
        '''
        Generate a random action.

        Returns:
            int: A random action.
        '''
        return int(np.random.choice(range(self.num_actions)))

    @staticmethod
    def softmax(x):
        '''
        Compute the softmax of an input vector x.

        Args:
            x (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Softmax of the input vector.
        '''
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):
        '''
        Normalize the input distribution x.

        Args:
            x (numpy.ndarray): Input distribution.

        Returns:
            numpy.ndarray: Normalized distribution.
        '''
        return np.dot(x, np.diag(1 / np.sum(x, 0)))


def create_ant(init_x, init_y, C):
    '''
    Create an ant with the specified initial position and custom prior C.

    Args:
        init_x (int): Initial X position.
        init_y (int): Initial Y position.
        C (numpy.ndarray): Custom prior matrix C.

    Returns:
        Ant: The created ant.
    '''
    A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
    B = np.zeros((cf.NUM_ACTIONS, cf.NUM_STATES, cf.NUM_STATES))
    for a in range(cf.NUM_ACTIONS):
        B[a, a, :] = 1.0
    mdp = MDP(A, B, C)
    ant = Ant(mdp, init_x, init_y)
    return ant


def dis(x1, y1, x2, y2):
    '''
    Calculate the Euclidean distance between two points.

    Args:
        x1 (float): X coordinate of the first point.
        y1 (float): Y coordinate of the first point.
        x2 (float): X coordinate of the second point.
        y2 (float): Y coordinate of the second point.

    Returns:
        float: Euclidean distance between the two points.
    '''
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def plot_path(path, save_name):
    '''
    Plot a path and save it as an image.

    Args:
        path (numpy.ndarray): Path to plot.
        save_name (str): Name of the image file to save.
    '''
    path = np.array(path)
    _, ax = plt.subplots(1, 1)
    ax.set_xlim(cf.GRID[0])
    ax.set_ylim(cf.GRID[1])
    ax.plot(path[:, 0], path[:, 1], "-o", color="red", alpha=0.4)
    plt.savefig(save_name)
    plt.close("all")


def save_gif(imgs, path, fps=32):
    '''
    Save a series of images as a GIF.

    Args:
        imgs (list): List of images to save as frames in the GIF.
        path (str): Path to save the GIF file.
        fps (int): Frames per second for the GIF.
    '''
    imageio.mimsave(path, imgs, fps=fps)


def main(num_steps, init_ants, max_ants, C, save=True, switch=False, name="", ant_only_gif=False):
    '''
    The main simulation function that runs the ant simulation.

    Args:
        num_steps (int): Number of simulation steps.
        init_ants (int): Number of initial ants.
        max_ants (int): Maximum number of ants.
        C (numpy.ndarray): Custom prior matrix C.
        save (bool): Flag to save images and GIF.
        switch (bool): Flag to switch food location.
        name (str): Name for saving simulation files.
        ant_only_gif (bool): Flag to generate an ant-only GIF.

    Returns:
        Tuple: A tuple containing the number of completed trips, list of paths, and total distance.
    '''
    # Initialize ants and set up the environment
    env = Env()
    ants = []
    paths = []

    for _ in range(init_ants):
        ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
        obs = env.get_obs(ant)
        A = env.get_A(ant)
        ant.mdp.set_A(A)
        ant.mdp.reset(obs)
        ants.append(ant)

    imgs = []
    completed_trips = 0
    distance = 0
    ant_locations = []
    round_trips_over_time = []

    for t in range(num_steps):
        t_dis = 0
        
        # Calculate total distance between ants
        for ant in ants:
            for ant_2 in ants:
                t_dis += dis(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)
        distance += t_dis / len(ants)

        # Display progress
        if t % (num_steps // 100) == 0:
            print(f"{t}/{num_steps}")

        # Add new ants periodically if the maximum number of ants has not been reached
        if t % cf.ADD_ANT_EVERY == 0 and len(ants) < max_ants:
            ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
            obs = env.get_obs(ant)
            A = env.get_A(ant)
            ant.mdp.set_A(A)
            ant.mdp.reset(obs)
            ants.append(ant)

        # Switch food location if required (halfway through the simulation)
        if switch and t % (num_steps // 2) == 0:
            # switch
            cf.FOOD_LOCATION[0] = cf.GRID[0] - cf.FOOD_LOCATION[0]

        # Iterate through ants
        for ant in ants:
            if not ant.is_returning:
                obs = env.get_obs(ant)
                A = env.get_A(ant)
                ant.mdp.set_A(A)
                action = ant.mdp.step(obs)
                env.step_forward(ant, action)
            else:
                is_complete, traj = env.step_backward(ant)
                completed_trips += int(is_complete)

                if is_complete:
                    paths.append(traj)

        # Decay the pheromones                    
        env.decay()

        # Save images and collect simulation metrics
        if save:
            if t in np.arange(0, num_steps, num_steps // 20):
                env.plot(ants, savefig=True, name=f"imgs/{name}_{t}.png")
            else:
                img = env.plot(ants, ant_only_gif=ant_only_gif)
                imgs.append(img)

        # Store round trips over time and ant locations
        round_trips_over_time.append(completed_trips / max_ants)
        ant_locations.append([[ant.x_pos, ant.y_pos] for ant in ants])

    """
    dis_coeff = 0
    for ant in ants:
        dis_coeff += sum(ant.distance)
    """

    # Save the animated gif of the simulation
    if save:
        save_gif(imgs, f"imgs/{name}.gif")

    ant_locations = np.array(ant_locations)
    round_trips_over_time = np.array(round_trips_over_time)
    np.save(f"imgs/{name}_locations", ant_locations)
    np.save(f"imgs/{name}_round_trips", round_trips_over_time)

    return completed_trips, np.array(paths), distance
