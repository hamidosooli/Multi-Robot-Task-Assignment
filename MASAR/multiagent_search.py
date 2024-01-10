from random import sample, choice, choices, seed
from scipy.stats import wrapcauchy, levy_stable
import numpy as np
import os
import sys
import decimal
from tqdm import tqdm
from record_vis import RecordVis

class Cell:
    def __init__(self, row, col, isboundary):
        self.visitors = []  # list of agents who have visited the cell
        self.notified = dict()  # list of agents who recognize the marking of that cell
        self.row = row
        self.col = col
        self.isboundary = isboundary

    def __repr__(self):
        return f"({self.row}, {self.col})"

class Map:
    ###### UPDATE WHATEVER IS DOING MAP CHECK STUFF ######
    def __init__(self, map_array): #map_size: int
        self.side_len = int(map_array.shape[0]) # Map array should be square
        self.map_size = self.side_len**2 # Map array should be square
        self.map = [[Cell(row=i, col=j, isboundary= map_array[i][j])
                     for j in range(self.side_len)]
                    for i in range(self.side_len)]
        self.cells_to_search = [self.map[i][j]
                                for i in range(self.side_len)
                                for j in range(self.side_len)
                                if self.map[i][j].isboundary == False]
        # self.map = np.array(self.map)
        self.visited_cells = 0

class Agent:
    def __init__(self, id_: int, v: float, seed_=None, fov="N/A"):
        self.id = id_
        self.v = v
        self.period = round(1 / v, 1)
        self.seed = seed_ if seed_ is not None else choice(range(sys.maxsize))
        self.search_time = 0
        self.fov = "N/A"


        # column where agents are indices and the value is the probability
        # that the agent will mark for them
        self.rel_row = None

    def __repr__(self):
        return f"Agent {self.id}"

    def place_in_map(self, map_):
        starting_cell = np.random.choice(map_.cells_to_search)
        return starting_cell.row, starting_cell.col


### Agent Acts on Map - Define Random Walk Agent
class RandomWalkAgent(Agent):
    def __init__(self, id_: int, v: float, fov=12, seed_=None, alpha = 2, rho =.02):
        super(RandomWalkAgent, self).__init__(id_, v, seed, fov=None)

        self.seed = seed_ if seed_ is not None else choice(range(sys.maxsize))
        self.fov = fov  # side length of observable square around the agent

        self.up, self.down, self.left, self.right = (-1, 0),  (1, 0), (0, -1), (0, 1)
        self.actions = [self.up,  self.down,  self.left, self.right]
        self.curr_pos = None
        self.prev_action = None

        # Levy Parameters
        self.alpha = alpha
        self.beta = 0
        self.scale_to_var = 1/(np.sqrt(2))
        self.levy_dist = levy_stable(self.alpha, self.beta, scale = self.scale_to_var)
        self.decision_wait_time = 0 # Relevent for Levy Walk

        # Turning Angle Parameters
        self.rho = rho # 0 is completely fair 1 is biased to previous action
        self.wrap_dist = wrapcauchy(self.rho)
        self.curr_angle = np.random.random()*np.pi*2
        self.dest_comb = 20
        self.path_to_take = None
        self.path_taken_counter = 0


    def account_for_boundary(self, map_):
        row, col = self.curr_pos
        actions = self.actions.copy()

        if row == 0:
            actions[0] = actions[1]
        elif row == map_.side_len - 1:
            actions[1] = actions[0]
        if col == 0:
            actions[2] = actions[3]
        elif col == map_.side_len - 1:
            actions[3] = actions[2]

        check_cells = [ (drow+row,dcol+col) for drow, dcol in [list(action) for action in actions]]

        for n, cell_ in enumerate(check_cells):
            m = (1,0,3,2) # Theres probably a better way to get opposite action
            if map_.map[cell_[0]][cell_[1]].isboundary:
                actions[n] = actions[m[n]]

        return actions

    def get_next_step(self, map_):
        """
        Randomly gets next step based on set of actions.
        Boundary conditions are reflective
        """
        prev_action = self.prev_action

        #Levy Walk each decision step continuos otherwise repeat last action
        if prev_action is None or self.decision_wait_time == 0:
            r = self.levy_dist.rvs()
            r = np.round(np.abs(r))
            self.decision_wait_time = int(r)

            r_angle = self.wrap_dist.rvs()
            self.curr_angle = (self.curr_angle + r_angle) % (2*np.pi)
            px = np.cos(self.curr_angle)
            py = np.sin(self.curr_angle)
            # print(px,py, r_angle)

            possible_actions = (int(-1*int(np.sign(py))),0), (0,int(np.sign(px)))
            px1 = abs(px)
            py1 = abs(py)
            # print(px1, py1, r_angle, self.curr_angle, self.decision_wait_time)
            self.path_to_take = [ possible_actions[i] for i in np.random.choice( [0,1], size = self.dest_comb, p = [py1**2, px1**2])]
        else:
            self.decision_wait_time -= 1


        actions_in_boundary = self.account_for_boundary(map_)

        desired_action = self.path_to_take[self.path_taken_counter]
        self.path_taken_counter = (1+self.path_taken_counter) % self.dest_comb

        if desired_action in actions_in_boundary:
            return desired_action
        else:
            # self.path_to_take = [( int(desired_action[0]*-1), int(desired_action[1]*-1)) if desired_action==x else x for x in self.path_to_take]
            if np.abs(desired_action[1]) == 1:
                dtheta  = self.curr_angle - np.pi/2
                if dtheta>0:
                    self.curr_angle = np.pi/2 - dtheta
                else:
                    self.curr_angle = np.pi/2 + np.abs(dtheta)

            elif np.abs(desired_action[0]) == 1:
                dtheta  = self.curr_angle - np.pi
                if dtheta>0:
                    self.curr_angle = np.pi - dtheta
                else:
                    self.curr_angle = np.pi + np.abs(dtheta)
            self.path_to_take = [ ( int(desired_action[0]*-1), int(desired_action[1]*-1)) if x == desired_action else x for x in self.path_to_take]
            # print("boundary hit", (self.curr_angle), np.cos(self.curr_angle), np.sin(self.curr_angle))
            # self.path_to_take = [ possible_actions[i] for i in np.random.choice( [0,1], size = self.dest_comb, p = [py1/(py1+px1),px1/(py1+px1)])]
            return ( int(desired_action[0]*-1) , int(desired_action[1]*-1) )

    def take_step(self, map_):
        # seed(self.seed)
        # prev_step = self.prev_pos
        # curr_step = self.curr_pos
        while True:
            if self.curr_pos is None:
                # This is the Initial step
                next_step = self.place_in_map(map_)
                row, col = next_step
            else:
                action = self.get_next_step(map_)
                # if action not in self.path_to_take:
                #     print(action, self.path_to_take,self.curr_angle)

                row, col = self.curr_pos[0] + action[0], self.curr_pos[1] + action[1]
                self.prev_action = action

            self.curr_pos= (row, col)

            cell = map_.map[row][col]
            if not cell.notified.get(self.id):
                cell.visitors.append(self.id)
                update = True
            else:
                update = False
            # self.update_cell(cell)
            yield cell, update


class MultiAgentSearch:
    def __init__(self, map_, agents, map_array,time_step=0.1, vis=False, sim_time = None, vis_outpath =None):
        self.map = map_
        self.agents = agents
        self.time_step = time_step
        self.vis = vis
        self.vis_outpath = vis_outpath
        self.sim_time = sim_time
        self.map_array = map_array

        self.period_dict = self.get_period_dict()
        self.steps_generators = [agent.take_step(map_) for agent in agents]

        self.record = {}

    def get_period_dict(self):
        period_dict = {}
        # every rate has a list of agents with that rate (1/v)
        for agent in self.agents:
            try:
                period_dict[agent.period].append(agent.id)
            except KeyError:
                period_dict[agent.period] = [agent.id]
        return period_dict

    def update_map(self, cell):
        global FINISHED
        # print(self.map.visited_cells)
        if self.map.visited_cells > len(self.map.cells_to_search)-1:
            FINISHED = True

        if len(cell.visitors) == 1:
            self.map.visited_cells += 1


    def divisible(self, num, den):
        if num < den:
            return False
        return round(float(decimal.Decimal(str(num)) % decimal.Decimal(str(den))), 1) == 0

    def search(self):
        global FINISHED
        self.counter = self.time_step
        while not FINISHED:

            if (self.sim_time != None) and (self.counter> self.sim_time):
                FINISHED = True

            # get ids of agents who will provide cells. These are indices in the steps generators object
            searchers_ids = []
            for period, agents_ids in self.period_dict.items():
                if self.divisible(self.counter, period):
                    searchers_ids.extend(agents_ids)

            searchers_ids = sample(searchers_ids, len(searchers_ids))  # shuffle

            for agent_id in searchers_ids:
                cell, update = next(self.steps_generators[agent_id])
                info = (agent_id, cell.row, cell.col)

                if self.vis:
                    try:
                        self.record[self.counter].append(info)
                    except KeyError:
                        self.record[self.counter] = [info]

                if update:
                    self.update_map(cell)
            self.counter = round(self.counter + self.time_step, 1)

                # if FINISHED:
        if self.vis:
            # Should instead write to file
            self.vis_record()
            return


    def vis_record(self):
        if self.vis_outpath == None:
            RecordVis().vis_record(self.record, self.map.side_len, self.map_array, 'vids/'+uuid.uuid1().hex)
        else:
            RecordVis().vis_record(self.record, self.map.side_len, self.map_array, 'vids/'+self.vis_outpath)


def get_agents(agent_class, num_agents, speeds, adj_matrix, fov, alpha, rho):
    """
    Get a number of agents and agents speeds and care adjacency matrix.
    Update each agent's care dict according to adj matrix
    return list of agents
    """
    agents = [agent_class(id_=i, v=speeds[i], fov=fov, seed_=i + 10, alpha = alpha, rho = rho) for i in range(num_agents)]
    # for agent in agents:
    #     agent.rel_row = adj_matrix[agent.id]
    return agents


def get_adj_str(adj_matrix):
    num_agents = len(adj_matrix)
    row_len = num_agents * 11
    top_row_len = num_agents * 11 - len(" Adjacency Matrix ")
    bot_row_len = row_len

    string = "*" * (top_row_len // 2) + " Adjacency Matrix " + "*" * (top_row_len // 2) + "\n"
    string += "\t\t" + "\t".join([f"Agent_{i}" for i in range(num_agents)])
    string += "\n"
    for idx, row in enumerate(adj_matrix):
        string += f"Agent_{idx}\t"
        string += "\t\t".join([str(n) for n in row])
        string += "\n\n"
    string += "*" * bot_row_len
    return string


def get_header(agents):
    header = ""
    for agent in agents:
        header += agent.__repr__()
        header += f". Speed: {agent.v} sq/s"
        header += "\n"
    header += "\n\n"
    return header


def output_results(round_, agents, adj_matrix, map_, out_path="results.csv"):
    """
    output a csv file with columns:
    search_time, Pi_1, Pi_2, Pi_3, etc
    """
    if not os.path.exists(out_path):
        header = get_header(agents)
        with open(out_path, "w") as out:
            out.write(header)
            cols = ",".join([f"Pi_{agent.id}" for agent in agents] + ["Search Time (s)\n"])
            out.write(cols)

    result = {agent.id: 0 for agent in agents}
    for i, row in enumerate(map_):
        for j, cell in enumerate(row):
            try:
                result[cell.visitors[0]] += 1
            except:
                # collapsed models
                continue

    with open(out_path, "a") as out:
        row = []
        search_time = round_.counter
        for agent_id in sorted(result.keys()):
            # search_time = round(agents[agent_id].search_time, 1)
            pi = round(result[agent_id] / search_time, 1)
            row.append(str(pi))
        row.append(str(search_time))
        out.write(",".join(row))
        out.write("\n")


# def main(agent_class, num_agents, fov, speeds, adj_matrix, map_size, out_path, vis):
def main(agent_class, num_agents, fov, speeds, adj_matrix, map_array, out_path, vis, sim_time, alpha, rho,vis_outpath):
    global FINISHED
    FINISHED = False
    agent_class = {0: Agent, 1: RandomWalkAgent}[agent_class]

    out_path = 'data/'+ out_path +".csv"
    agents = get_agents(agent_class, num_agents, speeds, adj_matrix, fov, alpha, rho)
    # test_map = np.zeros((int(map_size**0.5), int(map_size**0.5)), dtype= int)
    map_ = Map(map_array)

    round_ = MultiAgentSearch(map_=map_, agents=agents, map_array=map_array, time_step=0.1, vis=vis, sim_time = sim_time, vis_outpath = vis_outpath)
    round_.search()
    output_results(round_, agents, adj_matrix, map_.map, out_path=out_path)


if __name__ == '__main__':
    import argparse
    import uuid

    description = """
    simulate a multiagent map coverage using an input numpy adjacency matrix.
    get simulation results as a csv file for all agents' performance and final search time
    """
    id_ = uuid.uuid1().hex
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--agent_type', type=int, default=1,
                        help="type of agent: 0: teleporting; 1:  Random Walk")
    parser.add_argument('--fov', type=int, default=3,
                        help="agent field of view in Manhattan distance. Only for guided random walk agent")
    parser.add_argument('--adj_matrix', type=str, default='networks/2/C.npy',
                        help="path to .npy square adjacency matrix")
    parser.add_argument('--map_array_path', type=str, default=None,
                        help='Path to Map in .py form, defaults to 10x10 array')
    parser.add_argument('--map_array', type= str, default=None,
                        help='Custom map array')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='number of rounds played')
    parser.add_argument('--out_path', type=str, default=f"{id_}",
                        help='path of output csv file')
    parser.add_argument('--visualize', type=int, default=0,
                        help='1 to generate a video for the search')
    parser.add_argument('--vis_outpath', type=str, default=None,
                        help='outpath for video')
    parser.add_argument('--sim_time', type=float, default=None,
                        help='Total time for Simulation (agent_speed = 1 sq/s)')
    parser.add_argument('--num_agents', type=int, default=2,
                        help="Number of Agents")
    parser.add_argument('--alpha', type=float, default=2,
                        help="Levy Walk alpha")
    parser.add_argument('--rho', type=float, default=.01,
                        help="Persistance turning angle rho")



    args = parser.parse_args()
    agent_type = args.agent_type
    fov = args.fov
    agent_class = agent_type#{0: Agent, 1: RandomWalkAgent}[agent_type]
    adj_matrix = np.load(args.adj_matrix)
    num_agents = args.num_agents #adj_matrix.shape[0]
    if args.map_array_path is not None:
       map_array = np.load(args.map_array_path)
    elif args.map_array is None:
        map_array = np.zeros((10,10))
    else:
        map_array =  eval('np.array(' + args.map_array + ')')


    num_rounds = args.num_rounds
    out_path = args.out_path
    vis = args.visualize
    sim_time = args.sim_time
    alpha = args.alpha
    rho = args.rho
    vis_outpath = args.vis_outpath

    speeds = np.ones(num_agents)
    for _ in tqdm(range(num_rounds)):
        main(agent_class=agent_class,
             num_agents=num_agents,
             fov=fov,
             speeds=speeds,
             adj_matrix=adj_matrix,
             map_array=map_array,
             out_path=out_path,
             vis=vis,
             sim_time=sim_time,
             alpha = alpha,
             rho = rho,
             vis_outpath= vis_outpath)
