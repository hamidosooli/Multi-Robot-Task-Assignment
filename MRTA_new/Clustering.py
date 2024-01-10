import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import linear_sum_assignment
from a_star import AStarPlanner
import collections

def Clustering(num_clusters, task_list, walls_x, walls_y):
    """
    :param num_clusters: is equal to the number of regions/rooms in the environment
    :param task_list: list of the tasks in the environment
    :return:
    """
    num_rows = 20
    num_cols = 20
    tasks_pos = []
    for task in task_list:
        tasks_pos.append(task.pos)

    # k-means
    clusters_coord = kmeans(tasks_pos, num_clusters)[0]
    clusters = vq(tasks_pos, clusters_coord)[0]

    for cluster_idx, cluster in enumerate(clusters_coord):

        int_clusters_coord = [int(cluster[0]), int(cluster[1])]
        random_neighbor = [[min(num_rows-1, int_clusters_coord[0] + 1), int_clusters_coord[1]],
                           [max(0, int_clusters_coord[0] - 1), int_clusters_coord[1]],
                           [int_clusters_coord[0], min(num_cols-1, int_clusters_coord[1] + 1)],
                           [int_clusters_coord[0], max(0, int_clusters_coord[1] - 1)],
                           [min(num_rows-1, int_clusters_coord[0] + 1), min(num_cols-1, int_clusters_coord[1] + 1)],
                           [max(0, int_clusters_coord[0] - 1), max(0, int_clusters_coord[1] - 1)],
                           [min(num_rows-1, int_clusters_coord[0] + 1), max(0, int_clusters_coord[1] - 1)],
                           [max(0, int_clusters_coord[0] - 1), min(num_cols-1, int_clusters_coord[1] + 1)]]

        for idx in range(len(random_neighbor)):
            if tuple(random_neighbor[idx]) in list(zip(walls_x, walls_y)):
                continue
            else:
                int_clusters_coord = random_neighbor[idx]
                break

        clusters_coord[cluster_idx] = int_clusters_coord

    grid_size = 1  # [m]
    robot_radius = .5  # [m]
    a_star = AStarPlanner(walls_x, walls_y, grid_size, robot_radius)
    a_star_NW = AStarPlanner([], [], grid_size, robot_radius)
    # assign robot into the relevant cluster
    for idx, cluster_id in enumerate(clusters):
        task_list[idx].cluster_id = cluster_id
        rx, ry = a_star.planning(clusters_coord[cluster_id][0], clusters_coord[cluster_id][1],
                                 task_list[idx].pos[0], task_list[idx].pos[1])
        rx_NW, ry_NW = a_star_NW.planning(clusters_coord[cluster_id][0], clusters_coord[cluster_id][1],
                                 task_list[idx].pos[0], task_list[idx].pos[1])
        task_list[idx].cluster_dist = [[x, y] for x, y in zip(rx[::-1], ry[::-1])]
        task_list[idx].cluster_dist_NW = [[x, y] for x, y in zip(rx_NW[::-1], ry_NW[::-1])]
    return clusters, clusters_coord


def weight_calc(robot_list, task_list, clusters, num_clusters, psi=.95):
    '''
    :param robot_list: list of the robot objects (rescue team)
    :param task_list:
    :param clusters: list of  the clusters calculated by Clustering
    :param psi: Coefficient for fulfilled tasks
    :return: the cost that is the opposite of the number of tasks each robot can save at each cluster
    '''
    cost = np.zeros((len(robot_list), num_clusters))
    for robot in robot_list:
        for task_id, cluster_id in enumerate(clusters):

            if task_id in robot.tasks:
                caps = [cap in robot.capabilities for cap in task_list[task_id].capabilities]
                robot.w_rc[cluster_id] += (1-psi) * sum(caps)
            if task_id in robot.tasks_full:
                robot.w_rc[cluster_id] += psi * len(task_list[task_id].capabilities)

        cost[robot.id, :] = -robot.w_rc
    return cost


def ClstrAsgn(robot_list, task_list, clusters, clusters_coord, num_clusters):
    cost = weight_calc(robot_list, task_list, clusters, num_clusters)
    robots_opt, clusters_opt = linear_sum_assignment(cost)
    task_list_update = task_list.copy()
    for robot_idx, robot in enumerate(robot_list):
        for cluster_idx, cluster in enumerate(clusters):
            if cluster == clusters_opt[robot_idx]:
                if task_list[cluster_idx].id in robot.tasks:  # check for the ability to rescue
                    # Assign the tasks to the robot for rescue
                    # check for the wall in cluster
                    if len(task_list[cluster_idx].cluster_dist) <= len(task_list[cluster_idx].cluster_dist_NW):
                        robot.tasks_init.append(task_list[cluster_idx].id)
                        robot.tasks_init_dist.append(len(task_list[cluster_idx].cluster_dist))
                        caps = [cap in robot.capabilities for cap in task_list[cluster_idx].capabilities]

                        for cap_id, cap in enumerate(caps):
                            if cap:
                                task_list[cluster_idx].rescued[cap_id] = True

                        # Update the list of the remaining tasks and set the task's rescue flag True
                        task_list_update.remove(task_list[cluster_idx])

                # Send the robot to cluster's center
                robot.pos = clusters_coord[cluster]
        # Sort the assignment on the basis of distance to the cluster coordination
        robot.tasks_init = [task_id for _, task_id in sorted(zip(robot.tasks_init_dist, robot.tasks_init))]
        # check for the wall in cluster
        for task_id, task in enumerate(robot.tasks_init):
            if len(task_list[task].cluster_dist) > len(task_list[task].cluster_dist_NW):
                robot.tasks_init.remove(task)
                del robot.tasks_init_dist[task_id]

    return task_list_update
