import config as cf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio

from sklearn.cluster import DBSCAN
from esda.moran import Moran
<<<<<<< HEAD
from libpysal.weights import lat2W
from scipy.spatial.distance import cdist
=======
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

# matplotlib.use("Agg")
matplotlib.use("TkAgg")


class Ant(object):
    def __init__(self, mdp, init_x, init_y):
        self.mdp = mdp
        self.x_pos = init_x
        self.y_pos = init_y
        self.traj = [(init_x, init_y)]
        self.distance = []
        self.backward_step = 0
        self.is_returning = False
        self.timer = 0
        # self.time_since_last_round_trip = 0
        self.number_of_round_trips = 0

    def update_forward(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.traj.append((x_pos, y_pos))
        self.distance.append(dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))

    def update_backward(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.distance.append(dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))

class Env(object):
    def __init__(self):
        self.visit_matrix = np.zeros((cf.GRID[0], cf.GRID[1]))
        self.obs_matrix = np.zeros((cf.NUM_OBSERVATIONS, cf.GRID[0], cf.GRID[1]))
<<<<<<< HEAD
        # self.obs_matrix[0, :, :] = 1.0
        self.obs_matrix[0, :, :] = 0.0
=======
        self.obs_matrix[0, :, :] = 1.0
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

    def get_A(self, ant):
        A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
        for s in range(cf.NUM_STATES):
            delta = cf.ACTION_MAP[s]
            A[:, s] = self.obs_matrix[:, ant.x_pos + delta[0], ant.y_pos + delta[1]]
        return A

    def get_obs(self, ant):
        obs_vec = self.obs_matrix[:, ant.x_pos, ant.y_pos]
        return np.argmax(obs_vec)

    def check_food(self, x_pos, y_pos):
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
        delta = cf.ACTION_MAP[action]
        x_pos = np.clip(ant.x_pos + delta[0], 1, cf.GRID[0] - 2)
        y_pos = np.clip(ant.y_pos + delta[1], 1, cf.GRID[1] - 2)

        if self.check_food(x_pos, y_pos) and np.random.rand() < cf.NEST_FACTOR:
            ant.is_returning = True
            ant.backward_step = 0

        if self.check_walls(ant.x_pos, ant.y_pos, x_pos, y_pos):
            ant.update_forward(x_pos, y_pos)

        """
        if len(ant.traj) > cf.MAX_LEN:
            pos = ant.traj[0]
            ant.update_backward(pos[0], pos[1])
            ant.traj = []
        """

    def step_backward(self, ant):
        path_len = len(ant.traj)
        next_step = path_len - (ant.backward_step + 1)
        pos = ant.traj[next_step]
        ant.update_backward(pos[0], pos[1])

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
        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                if (curr_obs > 0) and (np.random.rand() < cf.DECAY_FACTOR):
                    curr_obs = curr_obs - 1
                    self.obs_matrix[:, x, y] = 0.0
                    self.obs_matrix[curr_obs, x, y] = 1.0

<<<<<<< HEAD
    # def get_pheromone_values(self):
    #     pheromone_values = []
    #     for x in range(cf.GRID[0]):
    #         for y in range(cf.GRID[1]):
    #             curr_obs = np.argmax(self.obs_matrix[:, x, y])
    #             pheromone_values.append(curr_obs)
    #     return np.array(pheromone_values)

    def get_nonzero_pheromone_locations(self):
        pheromone_locations = []
        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                # if self.obs_matrix[:, x, y] > 0.0:
                if curr_obs > 0.0:
                    # print(curr_obs)
                    pheromone_locations.append((x, y))
        return np.array(pheromone_locations)

    def get_values(self, ants):
        ant_locations = []
        pheromone_values = []
        for ant in ants:
            ant_locations.append((ant.x_pos, ant.y_pos))
            pheromone_values.append(np.argmax(self.obs_matrix[:, ant.x_pos, ant.y_pos]))
        return np.array(ant_locations), np.array(pheromone_values)
        
    def run_dbscan_on_pheromone_locs(self, eps, min_samples):
        pheromone_locations = self.get_nonzero_pheromone_locations()
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(pheromone_locations)
        return labels

    def run_dbscan_on_ant_locs(self, eps, min_samples, ants):
        ant_locations, _ = self.get_values(ants)
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(ant_locations)
        return labels

=======
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da
    def plot(self, ants, savefig=False, name="", ant_only_gif=False):
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

        # Overlay the green rectangle for FOOD_LOCATION
        food_x, food_y = cf.FOOD_LOCATION
        food_width, food_height = cf.FOOD_SIZE
        ax.add_patch(plt.Rectangle((food_x, food_y), food_width, food_height, color="green", alpha=0.5))

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
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.p0 = np.exp(-16)

        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.num_actions = self.B.shape[0]

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
        self.A = A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

    def reset(self, obs):
        self.t = 0
        self.curr_obs = obs
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.prev_action = self.random_action()

    def step(self, obs):
        """ state inference """
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.prev_action], self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(prior)

        """ action inference """
        SCALE = 10
        neg_efe = np.zeros([self.num_actions, 1])
        for a in range(self.num_actions):
            fs = np.dot(self.B[a], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)
            utility = np.sum(fo * np.log(fo / self.C), axis=0)
            utility = utility[0]
            neg_efe[a] -= utility / SCALE

        # priors
        neg_efe[4] -= 20.0
        neg_efe[cf.OPPOSITE_ACTIONS[self.prev_action]] -= 20.0  # type: ignore

        # action selection
        self.uQ = self.softmax(neg_efe)
        action = np.argmax(np.random.multinomial(1, self.uQ.squeeze()))
        self.prev_action = action
        return action

    def random_action(self):
        return int(np.random.choice(range(self.num_actions)))

    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):
        return np.dot(x, np.diag(1 / np.sum(x, 0)))


def create_ant(init_x, init_y, C):
    A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
    B = np.zeros((cf.NUM_ACTIONS, cf.NUM_STATES, cf.NUM_STATES))
    for a in range(cf.NUM_ACTIONS):
        B[a, a, :] = 1.0
    mdp = MDP(A, B, C)
    ant = Ant(mdp, init_x, init_y)
    return ant


def dis(x1, y1, x2, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def plot_path(path, save_name):
    path = np.array(path)
    _, ax = plt.subplots(1, 1)
    ax.set_xlim(cf.GRID[0])
    ax.set_ylim(cf.GRID[1])
    ax.plot(path[:, 0], path[:, 1], "-o", color="red", alpha=0.4)
    plt.savefig(save_name)
    plt.close("all")


def save_gif(imgs, path, fps=32):
    imageio.mimsave(path, imgs, fps=fps)

<<<<<<< HEAD
def main(num_steps, init_ants, max_ants, C, ctr, num_runs, save=True, switch=False, name="", ant_only_gif=False):
=======

def main(num_steps, init_ants, max_ants, C, save=True, switch=False, name="", ant_only_gif=False):
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da
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

<<<<<<< HEAD
    Morans_i_values = []
=======
    # Moran
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

    distance = 0
    distances = []

    cluster_density = 0
    cluster_densities = []
<<<<<<< HEAD
    num_clusters = []
=======
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

    ant_locations = []
    num_round_trips_per_time = []

    num_rt_at_time_t = 0

    for t in range(num_steps):
        t_dis = 0

        for ant in ants:
            for ant_2 in ants:
                t_dis += dis(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)

        current_avg_dist = t_dis / len(ants)
<<<<<<< HEAD
        distance += current_avg_dist
        distances.append(current_avg_dist)

        ant_positions, pheromone_values = env.get_values(ants)

        # Combine ant locations and pheromone values
        Z = np.column_stack((ant_positions, pheromone_values))

        # Create weight matrix
        w = lat2W(Z.shape[0], Z.shape[1])

        # Create PySAL Moran obj
        m_i = Moran(Z, w)

        # add to the Moran's I list
        Morans_i_values.append(m_i.I)

        if len(paths) == 0:
            cluster_densities.append(0)
            num_clusters.append(0)
        else:

            # ant_trajectory_locations = [point for trajectory in paths for point in trajectory]


            # Create and fit a DBSCAN model
            eps = 5 
            # min_samples = len(ants) // 5
            min_samples = 3
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(ant_positions)

            # the number of DBSCAN Clusters
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

            # Calculate cluster sizes
            cluster_sizes = [len(ant_positions[dbscan_labels == label]) for label in set(dbscan_labels)]

            # Calculate the average density
            if cluster_sizes:
                average_density = len(ant_positions) / len(cluster_sizes)
            else:
                average_density = 0

            # print("Average Density of Clusters:", average_density)

            cluster_densities.append(average_density)
            num_clusters.append(n_clusters)


        # if t % (num_steps // 100) == 0:
        print(f"Step {t + 1}/{num_steps} of Simulation {ctr + 1}/{num_runs}")
=======

        distance += current_avg_dist

        distances.append(current_avg_dist)

        ########################################################################
        if len(paths) == 0:
            cluster_densities.append(0)
        else:
            # DBSCAN Clustering
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
            # cluster_sizes = [np.sum(cluster_labels == label) for label in unique_clusters if label != -1]

            # Calculate stigmergy-related metrics based on cluster analysis
            # average_cluster_size = np.mean(cluster_sizes)
            if num_clusters > 0:
                cluster_density += len(data_points) / num_clusters  # Total deposits divided by the number of clusters
            else:
                cluster_density += 0

            cluster_densities.append(cluster_density)

            # # You can further analyze and visualize cluster properties to assess stigmergy
            # print(f"\ncluster_labels: {cluster_labels}")
            # print(f"unique_clusters: {unique_clusters}")
            # print(f"num_clusters: {num_clusters}")
            # print(f"cluster_sizes: {cluster_sizes}")
            # print(f"average_cluster_size: {average_cluster_size}")
            # print(f"cluster_density: {cluster_density}\n")

        ########################################################################

        # if t % (num_steps // 100) == 0:
        print(f"{t}/{num_steps}")
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

        if t % cf.ADD_ANT_EVERY == 0 and len(ants) < max_ants:
            ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
            obs = env.get_obs(ant)
            A = env.get_A(ant)
            ant.mdp.set_A(A)
            ant.mdp.reset(obs)
            ants.append(ant)

<<<<<<< HEAD
        # if switch and t % (num_steps // 2) == 0:
        #     cf.FOOD_LOCATION[0] = cf.GRID[0] - cf.FOOD_LOCATION[0]
=======
        if switch and t % (num_steps // 2) == 0:
            # switch
            cf.FOOD_LOCATION[0] = cf.GRID[0] - cf.FOOD_LOCATION[0]

        # num_rt_at_time_t = 0
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da

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

                    # increment the number of round trips by 1
                    ant.number_of_round_trips += 1
                    num_rt_at_time_t += 1

        num_round_trips_per_time.append(num_rt_at_time_t)
    
        if save:
<<<<<<< HEAD
            if t in np.arange(0, num_steps, num_steps // 20):
            # if t in np.arange(0, num_steps):
                env.plot(ants, savefig=True, name=f"my_imgs_full_sim/{name}_{t}.png")
=======
            # if t in np.arange(0, num_steps, num_steps // 20):
            if t in np.arange(0, num_steps, num_steps // 10):
                # env.plot(ants, savefig=True, name=f"imgs/{name}_{t}.png")
                # env.plot(ants, savefig=True, name=f"my_imgs/{name}_{t}.png")
                # env.plot(ants, savefig=True, name=f"my_imgs_2/{name}_{t}.png")
                # env.plot(ants, savefig=True, name=f"my_imgs_dbscan/{name}_{t}.png")
                # env.plot(ants, savefig=True, name=f"my_imgs_dbscan_2/{name}_{t}.png")
                env.plot(ants, savefig=True, name=f"my_imgs_dbscan_70_ants/{name}_{t}.png")
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da
            else:
                img = env.plot(ants, ant_only_gif=ant_only_gif)
                imgs.append(img)

        # round_trips_over_time.append(completed_trips / max_ants)
        ant_locations.append([[ant.x_pos, ant.y_pos] for ant in ants])

<<<<<<< HEAD
    if save:
        save_gif(imgs, f"my_imgs_full_sim/{name}.gif")


    ant_locations = np.array(ant_locations)

    np.save(f"my_imgs_full_sim/{name}_locations", ant_locations)


    return completed_trips, np.array(paths), distance, distances, num_round_trips_per_time, ants, cluster_densities, num_clusters, Morans_i_values

=======
    """
    dis_coeff = 0
    for ant in ants:
        dis_coeff += sum(ant.distance)
    """

    if save:
        # save_gif(imgs, f"imgs/{name}.gif")
        # save_gif(imgs, f"my_imgs_dbscan/{name}.gif")
        # save_gif(imgs, f"my_imgs_dbscan_2/{name}.gif")
        save_gif(imgs, f"my_imgs_dbscan_70_ants/{name}.gif")


    ant_locations = np.array(ant_locations)
    # round_trips_over_time = np.array(round_trips_over_time)

    # np.save(f"imgs/{name}_locations", ant_locations)
    # np.save(f"imgs/{name}_round_trips", round_trips_over_time)

    # np.save(f"my_imgs_dbscan/{name}_locations", ant_locations)
    # np.save(f"my_imgs_dbscan/{name}_round_trips", round_trips_over_time)

    # np.save(f"my_imgs_dbscan_2/{name}_locations", ant_locations)
    # np.save(f"my_imgs_dbscan_2/{name}_round_trips", round_trips_over_time)

    np.save(f"my_imgs_dbscan_70_ants/{name}_locations", ant_locations)
    # np.save(f"my_imgs_dbscan_70_ants/{name}_round_trips", round_trips_over_time)


    return completed_trips, np.array(paths), distance, distances, num_round_trips_per_time, ants, cluster_densities
>>>>>>> 3d30b4e04b3fdeb244d45b33e7d9e9c1dcf668da
