# def main(num_steps, init_ants, max_ants, C, save=True, switch=False, name="", ant_only_gif=False):
#     env = Env()
#     ants = []
#     paths = []
#     for _ in range(init_ants):
#         ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
#         obs = env.get_obs(ant)
#         A = env.get_A(ant)
#         ant.mdp.set_A(A)
#         ant.mdp.reset(obs)
#         ants.append(ant)

#     imgs = []
#     completed_trips = 0

#     # Moran

#     distance = 0
#     distances = []

#     cluster_density = 0
#     cluster_densities = []

#     ant_locations = []
#     num_round_trips_per_time = []

#     num_rt_at_time_t = 0

#     for t in range(num_steps):
#         t_dis = 0

#         for ant in ants:
#             for ant_2 in ants:
#                 t_dis += dis(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)

#         current_avg_dist = t_dis / len(ants)
#         distance += current_avg_dist
#         distances.append(current_avg_dist)

#         if len(paths) == 0:
#             cluster_densities.append(0)
#         else:

#             # a list of coordinates of locations in each ant path - used for DBSCAN Clustering and Moran's I
#             data_points = [point for trajectory in paths for point in trajectory]


#             ################################ DBSCAN ################################ 
#             # Convert data_points to a NumPy array
#             # data_points_array = np.array(data_points)

#             # Apply DBSCAN with appropriate parameters
#             eps = 5  # Adjust based on the characteristic distance between pheromone deposits
#             min_samples = 1  # Adjust based on the minimum number of deposits required to form a cluster

#             pheromone_locs = env.get_nonzero_pheromone_locations()

#             # dbscan_labels = env.run_dbscan_on_pheromone_locs(eps, min_samples)
#             ant_dbscan_labels = env.run_dbscan_on_ant_locs(eps, min_samples, ants)



#             # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             # cluster_labels = dbscan.fit_predict(data_points_array)

#             # Now 'cluster_labels' contains cluster assignments for each data point
#             # -1 indicates noise points, and other values represent cluster labels

#             # Analyze the clusters and measure stigmergy
#             unique_clusters = np.unique(ant_dbscan_labels)
#             num_clusters = len(unique_clusters) - 1  # Exclude noise cluster (-1)

#             # Measure cluster characteristics (e.g., size, density, centrality)
#             # cluster_sizes = [np.sum(cluster_labels == label) for label in unique_clusters if label != -1]

#             # Calculate stigmergy-related metrics based on cluster analysis
#             # average_cluster_size = np.mean(cluster_sizes)
#             if num_clusters > 0:
#                 cluster_density += len(pheromone_locs) / num_clusters  # Total deposits divided by the number of clusters
#             else:
#                 cluster_density += 0

#             cluster_densities.append(cluster_density)
#             ################################ DBSCAN ################################ 


#             ################################ Moran's I ################################
#             # # Extract X and Y coordinates
#             # path_x_coords, path_y_coords = zip(*data_points)

#             # # Example substance values (replace with your actual values)
#             # pheremone_values = np.arange(len(data_points))

#             # # Calculate Moran's I
#             # moran_i = morans_i(list(zip(path_x_coords, path_y_coords)), pheremone_values)
#             ################################ Moran's I ################################

#             # # You can further analyze and visualize cluster properties to assess stigmergy
#             # print(f"\ncluster_labels: {cluster_labels}")
#             # print(f"unique_clusters: {unique_clusters}")
#             # print(f"num_clusters: {num_clusters}")
#             # print(f"cluster_sizes: {cluster_sizes}")
#             # print(f"average_cluster_size: {average_cluster_size}")
#             # print(f"cluster_density: {cluster_density}\n")

#         # if t % (num_steps // 100) == 0:
#         print(f"{t}/{num_steps}")

#         if t % cf.ADD_ANT_EVERY == 0 and len(ants) < max_ants:
#             ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
#             obs = env.get_obs(ant)
#             A = env.get_A(ant)
#             ant.mdp.set_A(A)
#             ant.mdp.reset(obs)
#             ants.append(ant)

#         if switch and t % (num_steps // 2) == 0:
#             cf.FOOD_LOCATION[0] = cf.GRID[0] - cf.FOOD_LOCATION[0]

#         for ant in ants:
#             if not ant.is_returning:
#                 obs = env.get_obs(ant)
#                 A = env.get_A(ant)
#                 ant.mdp.set_A(A)
#                 action = ant.mdp.step(obs)
#                 env.step_forward(ant, action)
#             else:
#                 is_complete, traj = env.step_backward(ant)
#                 completed_trips += int(is_complete)

#                 if is_complete:
#                     paths.append(traj)

#                     # increment the number of round trips by 1
#                     ant.number_of_round_trips += 1
#                     num_rt_at_time_t += 1

#         num_round_trips_per_time.append(num_rt_at_time_t)
    
#         if save:
#             # if t in np.arange(0, num_steps, num_steps // 20):
#             if t in np.arange(0, num_steps, num_steps // 10):
#                 # env.plot(ants, savefig=True, name=f"imgs/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_2/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_dbscan/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_dbscan_2/{name}_{t}.png")
#                 env.plot(ants, savefig=True, name=f"my_imgs_dbscan_70_ants/{name}_{t}.png")
#             else:
#                 img = env.plot(ants, ant_only_gif=ant_only_gif)
#                 imgs.append(img)

#         # round_trips_over_time.append(completed_trips / max_ants)
#         ant_locations.append([[ant.x_pos, ant.y_pos] for ant in ants])

#     if save:
#         # save_gif(imgs, f"imgs/{name}.gif")
#         # save_gif(imgs, f"my_imgs_dbscan/{name}.gif")
#         # save_gif(imgs, f"my_imgs_dbscan_2/{name}.gif")
#         save_gif(imgs, f"my_imgs_dbscan_70_ants/{name}.gif")


#     ant_locations = np.array(ant_locations)
#     # round_trips_over_time = np.array(round_trips_over_time)

#     # np.save(f"imgs/{name}_locations", ant_locations)
#     # np.save(f"imgs/{name}_round_trips", round_trips_over_time)

#     # np.save(f"my_imgs_dbscan/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan/{name}_round_trips", round_trips_over_time)

#     # np.save(f"my_imgs_dbscan_2/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan_2/{name}_round_trips", round_trips_over_time)

#     np.save(f"my_imgs_dbscan_70_ants/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan_70_ants/{name}_round_trips", round_trips_over_time)


#     return completed_trips, np.array(paths), distance, distances, num_round_trips_per_time, ants, cluster_densities


# def main(num_steps, init_ants, max_ants, C, save=True, switch=False, name="", ant_only_gif=False):
#     env = Env()
#     ants = []
#     paths = []
#     for _ in range(init_ants):
#         ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
#         obs = env.get_obs(ant)
#         A = env.get_A(ant)
#         ant.mdp.set_A(A)
#         ant.mdp.reset(obs)
#         ants.append(ant)

#     imgs = []
#     completed_trips = 0

#     # Moran

#     distance = 0
#     distances = []

#     cluster_density = 0
#     cluster_densities = []

#     ant_locations = []
#     num_round_trips_per_time = []

#     num_rt_at_time_t = 0

#     for t in range(num_steps):
#         t_dis = 0

#         for ant in ants:
#             for ant_2 in ants:
#                 t_dis += dis(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)

#         current_avg_dist = t_dis / len(ants)

#         distance += current_avg_dist

#         distances.append(current_avg_dist)

#         if len(paths) == 0:
#             cluster_densities.append(0)
#         else:

#             # a list of coordinates of locations in each ant path - used for DBSCAN Clustering and Moran's I
#             data_points = [point for trajectory in paths for point in trajectory]


#             ################################ DBSCAN ################################ 
#             # Convert data_points to a NumPy array
#             data_points_array = np.array(data_points)

#             # Apply DBSCAN with appropriate parameters
#             eps = 0.1  # Adjust based on the characteristic distance between pheromone deposits
#             min_samples = 5  # Adjust based on the minimum number of deposits required to form a cluster

#             dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#             cluster_labels = dbscan.fit_predict(data_points_array)

#             # Now 'cluster_labels' contains cluster assignments for each data point
#             # -1 indicates noise points, and other values represent cluster labels

#             # Analyze the clusters and measure stigmergy
#             unique_clusters = np.unique(cluster_labels)
#             num_clusters = len(unique_clusters) - 1  # Exclude noise cluster (-1)

#             # Measure cluster characteristics (e.g., size, density, centrality)
#             # cluster_sizes = [np.sum(cluster_labels == label) for label in unique_clusters if label != -1]

#             # Calculate stigmergy-related metrics based on cluster analysis
#             # average_cluster_size = np.mean(cluster_sizes)
#             if num_clusters > 0:
#                 cluster_density += len(data_points) / num_clusters  # Total deposits divided by the number of clusters
#             else:
#                 cluster_density += 0

#             cluster_densities.append(cluster_density)
#             ################################ DBSCAN ################################ 


#             ################################ Moran's I ################################
#             # # Extract X and Y coordinates
#             # path_x_coords, path_y_coords = zip(*data_points)

#             # # Example substance values (replace with your actual values)
#             # pheremone_values = np.arange(len(data_points))

#             # # Calculate Moran's I
#             # moran_i = morans_i(list(zip(path_x_coords, path_y_coords)), pheremone_values)
#             ################################ Moran's I ################################

#             # # You can further analyze and visualize cluster properties to assess stigmergy
#             # print(f"\ncluster_labels: {cluster_labels}")
#             # print(f"unique_clusters: {unique_clusters}")
#             # print(f"num_clusters: {num_clusters}")
#             # print(f"cluster_sizes: {cluster_sizes}")
#             # print(f"average_cluster_size: {average_cluster_size}")
#             # print(f"cluster_density: {cluster_density}\n")

#         # if t % (num_steps // 100) == 0:
#         print(f"{t}/{num_steps}")

#         if t % cf.ADD_ANT_EVERY == 0 and len(ants) < max_ants:
#             ant = create_ant(cf.INIT_X, cf.INIT_Y, C)
#             obs = env.get_obs(ant)
#             A = env.get_A(ant)
#             ant.mdp.set_A(A)
#             ant.mdp.reset(obs)
#             ants.append(ant)

#         if switch and t % (num_steps // 2) == 0:
#             # switch
#             cf.FOOD_LOCATION[0] = cf.GRID[0] - cf.FOOD_LOCATION[0]

#         # num_rt_at_time_t = 0

#         for ant in ants:
#             if not ant.is_returning:
#                 obs = env.get_obs(ant)
#                 A = env.get_A(ant)
#                 ant.mdp.set_A(A)
#                 action = ant.mdp.step(obs)
#                 env.step_forward(ant, action)
#             else:
#                 is_complete, traj = env.step_backward(ant)
#                 completed_trips += int(is_complete)

#                 if is_complete:
#                     paths.append(traj)

#                     # increment the number of round trips by 1
#                     ant.number_of_round_trips += 1
#                     num_rt_at_time_t += 1

#         num_round_trips_per_time.append(num_rt_at_time_t)
    
#         if save:
#             # if t in np.arange(0, num_steps, num_steps // 20):
#             if t in np.arange(0, num_steps, num_steps // 10):
#                 # env.plot(ants, savefig=True, name=f"imgs/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_2/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_dbscan/{name}_{t}.png")
#                 # env.plot(ants, savefig=True, name=f"my_imgs_dbscan_2/{name}_{t}.png")
#                 env.plot(ants, savefig=True, name=f"my_imgs_dbscan_70_ants/{name}_{t}.png")
#             else:
#                 img = env.plot(ants, ant_only_gif=ant_only_gif)
#                 imgs.append(img)

#         # round_trips_over_time.append(completed_trips / max_ants)
#         ant_locations.append([[ant.x_pos, ant.y_pos] for ant in ants])

#     if save:
#         # save_gif(imgs, f"imgs/{name}.gif")
#         # save_gif(imgs, f"my_imgs_dbscan/{name}.gif")
#         # save_gif(imgs, f"my_imgs_dbscan_2/{name}.gif")
#         save_gif(imgs, f"my_imgs_dbscan_70_ants/{name}.gif")


#     ant_locations = np.array(ant_locations)
#     # round_trips_over_time = np.array(round_trips_over_time)

#     # np.save(f"imgs/{name}_locations", ant_locations)
#     # np.save(f"imgs/{name}_round_trips", round_trips_over_time)

#     # np.save(f"my_imgs_dbscan/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan/{name}_round_trips", round_trips_over_time)

#     # np.save(f"my_imgs_dbscan_2/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan_2/{name}_round_trips", round_trips_over_time)

#     np.save(f"my_imgs_dbscan_70_ants/{name}_locations", ant_locations)
#     # np.save(f"my_imgs_dbscan_70_ants/{name}_round_trips", round_trips_over_time)


#     return completed_trips, np.array(paths), distance, distances, num_round_trips_per_time, ants, cluster_densities
