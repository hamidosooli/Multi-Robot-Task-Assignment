"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
from gridworld_multi_agent1_1 import animate
show_animation = False
from victim import Victim

capabilities = ['FirstAids', 'DebrisRemover', 'OxygenCylinder', 'Defuser', 'Manipulator', 'FireExtinguisher']
make_span = {'FirstAids': 15, 'DebrisRemover': 30, 'OxygenCylinder': 20,
             'Defuser': 10, 'Manipulator': 45, 'FireExtinguisher': 35}

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..", start_node)
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        if len(ox) == 0 and len(oy) == 0:
            self.min_x = 0
            self.min_y = 0
            self.max_x = 19
            self.max_y = 19
            print("min_x:", self.min_x)
            print("min_y:", self.min_y)
            print("max_x:", self.max_x)
            print("max_y:", self.max_y)

            self.x_width = round((self.max_x - self.min_x) / self.resolution)
            self.y_width = round((self.max_y - self.min_y) / self.resolution)
            print("x_width:", self.x_width)
            print("y_width:", self.y_width)

            self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool).tolist()

            for ix in range(self.x_width):
                x = self.calc_grid_position(ix, self.min_x)
                for iy in range(self.y_width):
                    y = self.calc_grid_position(iy, self.min_y)
        else:
            self.min_x = round(min(ox))
            self.min_y = round(min(oy))
            self.max_x = round(max(ox))
            self.max_y = round(max(oy))
            print("min_x:", self.min_x)
            print("min_y:", self.min_y)
            print("max_x:", self.max_x)
            print("max_y:", self.max_y)

            self.x_width = round((self.max_x - self.min_x) / self.resolution)
            self.y_width = round((self.max_y - self.min_y) / self.resolution)
            print("x_width:", self.x_width)
            print("y_width:", self.y_width)

            # obstacle map generation
            self.obstacle_map = [[False for _ in range(self.y_width)]
                                 for _ in range(self.x_width)]
            for ix in range(self.x_width):
                x = self.calc_grid_position(ix, self.min_x)
                for iy in range(self.y_width):
                    y = self.calc_grid_position(iy, self.min_y)
                    for iox, ioy in zip(ox, oy):
                        d = math.hypot(iox - x, ioy - y)
                        if d <= self.rr:
                            self.obstacle_map[ix][iy] = True
                            break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]#,
                  # [-1, -1, math.sqrt(2)],
                  # [-1, 1, math.sqrt(2)],
                  # [1, -1, math.sqrt(2)],
                  # [1, 1, math.sqrt(2)]]
        return motion



def main():
    print(__file__ + " start!!")
    exp_name = 'FindSurvivors'

    plt.rcParams.update({'font.size': 22})

    with h5py.File(f'MRTA.hdf5', 'r') as f:
        num_robots = np.asarray(f[f'RS_size']).tolist()
        num_victims = np.asarray(f[f'Victims_size']).tolist()
        starts = np.asarray(f[f'RS_starts']).tolist()
        travel2clusters = np.asarray(f[f'RS_travel2clusters']).tolist()
        travel2clusters = [[int(x), int(y)] for x, y in travel2clusters]
        # unnecessary floor puts cluster center in the wall!!!!!!!!!!!
        travel2clusters[1][1] += 1
        tasks = []
        for i in range(num_robots):
            list_t = []
            s1 = np.asarray(f[f'RS{i}_Step_1']).tolist()
            list_t.append(s1)
            s2 = np.asarray(f[f'RS{i}_Step_2']).tolist()
            list_t.append(s2)
            s3 = np.asarray(f[f'RS{i}_Step_3']).tolist()
            list_t.append(s3)
            tasks.append(list_t)

    file_name = f'../MASAR/multi_agent_Q_learning_{exp_name}.hdf5'

    grid_size = 1.0  # [m]
    robot_radius = .50  # [m]
    env_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])

    # env_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 1, 1, 1, 1, 1, 1, 0],
    #                     [0, 1, 0, 0, 0, 0, 0, 0],
    #                     [0, 1, 0, 0, 0, 0, 0, 1],
    #                     [0, 1, 0, 0, 0, 0, 0, 1],
    #                     [0, 0, 0, 0, 0, 0, 0, 1],
    #                     [0, 0, 1, 1, 1, 1, 1, 1],
    #                     [0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 1, 1, 1, 1, 1, 1],
    #                     [0, 0, 0, 0, 0, 0, 0, 1],
    #                     [1, 0, 0, 0, 1, 0, 0, 1],
    #                     [1, 0, 0, 0, 1, 0, 0, 1],
    #                     [0, 0, 0, 0, 1, 0, 0, 1],
    #                     [1, 1, 1, 1, 1, 0, 0, 0]])

    # set obstacle positions
    oy, ox = np.where(env_map == 1)
    # oy = oy

    # sy, sx = starts[0]
    # gy, gx = travel2clusters[0]
    # a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    # rx, ry = a_star.planning(sx, sy, gx, gy)

    # if show_animation:  # pragma: no cover
    #     plt.plot(ox, oy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")
    # rx, ry = a_star.planning(sx, sy, gx, gy)
    #
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    # start and goal position
    rescue_team_Traj = [[] for _ in range(num_robots)]
    Roles = [b'r' for _ in range(num_robots)]
    VFDs = [0 for _ in range(num_robots)]

    for num in range(num_robots):
        temp_task = []
        for tsk in tasks[num]:
            temp_task += tsk
            tasks[num] = temp_task
    v0 = Victim(0, [16., 7.], make_span, ['DebrisRemover', 'OxygenCylinder'], capabilities)
    v1 = Victim(1, [15., 12.], make_span, ['Defuser', 'Manipulator'], capabilities)
    v2 = Victim(2, [6., 5.], make_span, ['DebrisRemover', 'FireExtinguisher'], capabilities)
    v3 = Victim(3, [11., 4.], make_span, ['OxygenCylinder', 'Manipulator'], capabilities)
    v4 = Victim(4, [0., 1.], make_span, ['FirstAids', 'DebrisRemover'], capabilities)
    v5 = Victim(5, [14., 14.], make_span, ['Manipulator', 'FireExtinguisher'], capabilities)
    v6 = Victim(6, [14., 12.], make_span, ['FirstAids', 'Manipulator'], capabilities)
    v7 = Victim(7, [3., 16.], make_span, ['FirstAids', 'Defuser'], capabilities)
    v8 = Victim(8, [10., 15.], make_span, ['DebrisRemover', 'Defuser'], capabilities)
    v9 = Victim(9, [0., 12.], make_span, ['FirstAids', 'Manipulator'], capabilities)
    victims = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]
    with h5py.File(file_name, 'r') as f:
        for num in range(num_robots):
            sy, sx = starts[num]
            gx, gy = travel2clusters[num]
            for tsk in tasks[num]:

                rx, ry = a_star.planning(sx, sy, gx, gy)
                temp = [[y, x] for x, y in zip(rx[::-1], ry[::-1])]
                for pos in temp:
                    rescue_team_Traj[num].append(pos)

                sy, sx = gy, gx
                gy, gx = victims[tsk].pos#f[f'victim{tsk}_trajectory'][0]  # [m]

    len_max = len(rescue_team_Traj[0])
    for num in range(num_robots):
        if len(rescue_team_Traj[num]) > len_max:
            len_max = len(rescue_team_Traj[num])
    for num in range(num_robots):
        while len(rescue_team_Traj[num]) < len_max:
            rescue_team_Traj[num].append(rescue_team_Traj[num][-1])

    victims_Traj = []
    with h5py.File(file_name, 'r') as f:
        for idx in range(num_victims):
            victims_Traj.append([victims[idx].pos])
            # victims_Traj.append(np.asarray(f[f'victim{idx}_trajectory']).tolist())
            while len(victims_Traj[idx]) < len_max:
                victims_Traj[idx].append(victims_Traj[idx][-1])
            if len(victims_Traj[idx]) > len_max:
                victims_Traj[idx] = victims_Traj[idx][:len_max]
    rescue_team_VFD_status = [np.ones((num_robots, 1, 1), dtype=bool) for _ in range(len_max)]

    animate(np.asarray(rescue_team_Traj), np.asarray(victims_Traj),
            np.asarray(VFDs), rescue_team_VFD_status, Roles, env_map, wait_time=0.5)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
