from scipy.stats import wrapcauchy, levy_stable
import numpy as np


class SearchAlgorithms:
    def __init__(self, max_vfd, init_pos, num_actions, num_rows, num_cols):
        self.action = None
        self.max_VisualField = max_vfd
        self.wereHere = np.ones((num_rows, num_cols))

        self.init_pos = init_pos
        self.curr_Pos = self.init_pos
        self.old_Pos = self.curr_Pos

        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.up, self.down, self.left, self.right = (-1, 0),  (1, 0), (0, -1), (0, 1)
        self.actions = [self.up,  self.down,  self.left, self.right]
        # Levy Parameters
        self.prev_action = None
        self.alpha = 0.8
        self.beta = 0
        self.scale_to_var = 1/(np.sqrt(2))
        self.levy_dist = levy_stable(self.alpha, self.beta, scale=self.scale_to_var)
        self.decision_wait_time = 0

        # Turning Angle Parameters
        self.rho = 0.1  # 0 is completely fair 1 is biased to previous action
        self.wrap_dist = wrapcauchy(self.rho)
        self.curr_angle = np.random.random()*np.pi*2
        self.dest_comb = 20
        self.path_to_take = None
        self.path_taken_counter = 0

    def straight_move(self, idx, were_here, env_map):
        """
        takes index to see if the agent is in search mode
        wereHere is a matrix that tracks the visited cells
        """
        if idx == (2 * self.max_VisualField + 1) ** 2:
            if len(np.argwhere(were_here)) > 0:
                for loc in np.argwhere(were_here):
                    if np.sqrt((loc[0] - self.old_Pos[0]) ** 2 + (loc[1] - self.old_Pos[1]) ** 2) == 1:
                        next_loc = loc
                        self.wereHere[self.curr_Pos[0], self.curr_Pos[1]] = 0

                        if env_map[next_loc[0], next_loc[1]] == 0:
                            self.curr_Pos = next_loc
                        break
                    else:
                        continue

    def random_walk(self, idx, pos, speed, env_map):
        """
        takes current location and its relevant index in the Q table
        as well as agents speed
        If the index is for the search mode,
         it randomly selects one of the 4 possible directions
         """
        row_lim = self.num_rows - 1
        col_lim = self.num_cols - 1
        row = pos[0]
        col = pos[1]
        if idx == (2 * self.max_VisualField + 1) ** 2:
            self.action = np.random.randint(self.num_actions)

            if self.action == 0:  # up
                next_loc = [max(row - speed, 0), col]
            elif self.action == 1:  # down
                next_loc = [min(row + speed, row_lim), col]
            elif self.action == 2:  # right
                next_loc = [row, min(col + speed, col_lim)]
            elif self.action == 3:  # left
                next_loc = [row, max(col - speed, 0)]

            if env_map[next_loc[0], next_loc[1]] == 0:
                self.curr_Pos = next_loc

    # The following three methods were developed by Mohamed Martini
    def get_nearby_location_visits(self, grid_cells):
        """ takes the grid of cells (represented by a 2D numpy array)
            returns a  of the locations (as x,y tuples) which are 1 unit away
            (ie: is UP, DOWN, LEFT, RIGHT of current agent position) together with their
            visit count which is an integer representing the number of times the cell has been visited
        """
        nearby_cell_visits = list()
        for row in range(0, len(grid_cells)):
            for col in range(0, len(grid_cells[row, :])):
                visit_num = grid_cells[row, col]
                loc = [row, col]
                if np.linalg.norm(np.subtract(loc, self.old_Pos)) == 1:
                    loc_visits = [loc, visit_num]
                    nearby_cell_visits.append(loc_visits)
        return nearby_cell_visits

    def get_minimum_visited_cells(self, location_visits):
        """ takes a list of tuples whose elements represent locations in the grid world together
            with their visit counts and returns an array of locations which have the minimum number
            of visits
        """
        min_visits = np.inf  # or any very large number (greater than any expected visit count)
        min_visited_locations = []
        # find the minimum visited number for cells corresponding with the passed locations
        for loc_visits in location_visits:
            times_visited = loc_visits[1]
            if times_visited < min_visits:
                min_visits = times_visited
        # filter the locations corresponding with this minimum visit number
        for loc in location_visits:
            if loc[1] == min_visits:
                min_visited_locations.append(loc)
        return min_visited_locations

    def ant_colony_move(self, cells_visited, idx, env_map):
        """ takes a 2D array representing the visit count for cells in the grid world
            and increments the current agents position toward the least visited neighboring cell
        """
        # increment the cell visit number
        cells_visited[self.old_Pos[0], self.old_Pos[1]] += 1
        if idx == (2 * self.max_VisualField + 1) ** 2:
            nearby_location_visits = self.get_nearby_location_visits(cells_visited)
            least_visited_locations = self.get_minimum_visited_cells(nearby_location_visits)
            # select a random location from the least visit locations nearby
            next_loc_ind = np.random.randint(0, len(least_visited_locations))
            next_loc = least_visited_locations[next_loc_ind][0]
            if env_map[next_loc[0], next_loc[1]] == 0:
                self.curr_Pos = next_loc

    # The following method was developed by Fernando Mazzoni
    def account_for_boundary(self):
        row, col = self.curr_Pos
        actions = self.actions.copy()

        if row == 0:
            actions[0] = actions[1]
        elif row == self.num_rows - 1:
            actions[1] = actions[0]
        if col == 0:
            actions[2] = actions[3]
        elif col == self.num_cols - 1:
            actions[3] = actions[2]

        check_cells = [[drow+row, dcol+col] for drow, dcol in [list(action) for action in actions]]

        for n, cell_ in enumerate(check_cells):
            m = [1, 0, 3, 2]  # get the opposite action
            if ((cell_[0] == 0 or cell_[0] == self.num_cols - 1) or
                (cell_[1] == 0 or cell_[1] == self.num_rows - 1)):
                actions[n] = actions[m[n]]

        return actions

    def levy_walk(self, env_map):
        """
        Randomly gets next step based on set of actions.
        Boundary conditions are reflective
        """
        prev_action = self.prev_action

        # Levy Walk each decision step continues otherwise repeat last action
        if prev_action is None or self.decision_wait_time == 0:
            r = self.levy_dist.rvs()
            r = np.round(np.abs(r))
            self.decision_wait_time = int(r)

            r_angle = self.wrap_dist.rvs()
            self.curr_angle = (self.curr_angle + r_angle) % (2*np.pi)
            px = np.cos(self.curr_angle)
            py = np.sin(self.curr_angle)

            possible_actions = [[int(-1*int(np.sign(py))), 0], [0, int(np.sign(px))]]
            px1 = abs(px)
            py1 = abs(py)
            self.path_to_take = [possible_actions[i] for i in np.random.choice([0, 1],
                                                                               size=self.dest_comb,
                                                                               p=[py1**2, px1**2])]
        else:
            self.decision_wait_time -= 1

        actions_in_boundary = self.account_for_boundary()
        desired_action = self.path_to_take[self.path_taken_counter]
        self.path_taken_counter = (1+self.path_taken_counter) % self.dest_comb

        if desired_action in actions_in_boundary:
            action = desired_action
        else:
            if np.abs(desired_action[1]) == 1:
                dtheta = self.curr_angle - np.pi/2
                if dtheta > 0:
                    self.curr_angle = np.pi/2 - dtheta
                else:
                    self.curr_angle = np.pi/2 + np.abs(dtheta)
            elif np.abs(desired_action[0]) == 1:
                dtheta = self.curr_angle - np.pi
                if dtheta > 0:
                    self.curr_angle = np.pi - dtheta
                else:
                    self.curr_angle = np.pi + np.abs(dtheta)
            self.path_to_take = [(int(desired_action[0]*-1), int(desired_action[1]*-1))
                                 if x == desired_action else x for x in self.path_to_take]

            action = [int(desired_action[0]*-1), int(desired_action[1]*-1)]

        next_loc = [self.curr_Pos[0] + action[0], self.curr_Pos[1] + action[1]]
        self.prev_action = action
        if env_map[next_loc[0], next_loc[1]] == 0:
            self.curr_Pos = next_loc

