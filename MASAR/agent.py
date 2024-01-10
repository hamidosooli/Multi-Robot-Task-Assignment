import numpy as np
from search_algorithms import SearchAlgorithms


class Agent(SearchAlgorithms):
    def __init__(self, agent_id, role, vfd, max_vfd, speed, init_pos, num_actions, num_rows, num_cols):
        super(Agent, self).__init__(max_vfd, init_pos, num_actions, num_rows, num_cols)
        self.Role = role  # can be 'r': rescuer, 's': scout, 'rs': rescuer and scout, 'v': victim
        self.id = agent_id  # an identification for the agent
        self.VisualField = vfd
        self.max_VisualField = max_vfd
        self.vfd_status = np.ones((2 * self.VisualField + 1, 2 * self.VisualField + 1), dtype=bool)

        self.curr_Sensation = [np.nan, np.nan]
        self.old_Sensation = self.curr_Sensation

        self.curr_Index = None
        self.old_Index = self.curr_Index

        self.CanSeeIt = False
        self.Finish = False
        self.Convergence = False
        self.First = True
        self.wereHere = np.ones((num_rows, num_cols))
        self.Speed = speed  # is the number of cells the agent can move in one time-step

        self.init_pos = init_pos
        self.curr_Pos = self.init_pos
        self.old_Pos = self.curr_Pos

        self.Traj = []  # Trajectory of the agent locations
        self.VFD_status_history = []
        self.RewHist = []
        self.RewHist_seen = []
        self.RewSum = []  # Keeps track of the rewards in each step
        self.RewSum_seen = []  # Keeps track of the rewards after receiving first data
        self.Steps = []  # Keeps track of the steps in each step
        self.Steps_seen = []  # Keeps track of the steps after receiving first data

        self.num_actions = num_actions
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.t_step_seen = 0
        self.action = None
        self.reward = None
        self.probs = np.nan
        self.Q = np.zeros(((2 * self.max_VisualField + 1) ** 2 + 1, self.num_actions))
        self.Q_hist = self.Q

    def reset(self):
        self.old_Pos = self.init_pos
        self.curr_Pos = self.init_pos
        self.old_Sensation = [np.nan, np.nan]
        self.curr_Sensation = [np.nan, np.nan]
        self.CanSeeIt = False
        self.Finish = False
        self.Convergence = False
        self.First = True
        self.t_step_seen = 0
        self.RewHist = []
        self.RewHist_seen = []
        self.Traj = []
        self.VFD_status_history = []
        self.wereHere = np.ones_like(self.wereHere)
        self.vfd_status = np.ones((2 * self.VisualField + 1, 2 * self.VisualField + 1), dtype=bool)

    def update_vfd(self, env_map):
        # rescuer visual field depth
        self.vfd_status = np.ones((2 * self.VisualField + 1, 2 * self.VisualField + 1), dtype=bool)
        vfd_j = 0
        for j in range(int(max(self.old_Pos[1] - self.VisualField, 0)),
                       int(min(self.num_cols, self.old_Pos[1] + self.VisualField + 1))):
            vfd_i = 0
            for i in range(int(max(self.old_Pos[0] - self.VisualField, 0)),
                           int(min(self.num_rows, self.old_Pos[0] + self.VisualField + 1))):
                if env_map[i, j] == 1:
                    self.vfd_status[vfd_i, vfd_j] = False
                    # Down or Up
                    if i - self.old_Pos[0] > 0 and j - self.old_Pos[1] == 0:
                        self.vfd_status[vfd_i:, vfd_j] = False
                    elif i - self.old_Pos[0] < 0 and j - self.old_Pos[1] == 0:
                        self.vfd_status[:min(vfd_i+1, self.VisualField), vfd_j] = False
                    # Right or Left
                    elif i - self.old_Pos[0] == 0 and j - self.old_Pos[1] > 0:
                        self.vfd_status[vfd_i, vfd_j:] = False
                    elif i - self.old_Pos[0] == 0 and j - self.old_Pos[1] < 0:
                        self.vfd_status[vfd_i, :min(vfd_j+1, self.VisualField)] = False
                    # Northeast or Northwest
                    elif i - self.old_Pos[0] < 0 and j - self.old_Pos[1] > 0:
                        self.vfd_status[:min(vfd_i+1, self.VisualField), vfd_j:] = False
                    elif i - self.old_Pos[0] < 0 and j - self.old_Pos[1] < 0:
                        self.vfd_status[:min(vfd_i+1, self.VisualField), :min(vfd_j+1, self.VisualField)] = False
                    # Southeast or Southwest
                    elif i - self.old_Pos[0] > 0 and j - self.old_Pos[1] > 0:
                        self.vfd_status[vfd_i:, vfd_j:] = False
                    elif i - self.old_Pos[0] > 0 and j - self.old_Pos[1] < 0:
                        self.vfd_status[vfd_i:, :min(vfd_j+1, self.VisualField)] = False

                vfd_i += 1
            vfd_j += 1

    def update_sensation(self, index, raw_sensation, sensation_evaluate, pos2pos, net_adj_mat, adj_mat):

        next_sensation = [np.nan, np.nan]
        self.CanSeeIt = False

        if any(sensation_evaluate[index, :]):
            which_victim = np.argwhere(sensation_evaluate[index, :])[0][0]
            for victim in np.argwhere(sensation_evaluate[index, :])[0]:
                if (np.linalg.norm(raw_sensation[index, victim, :]) <
                    np.linalg.norm(raw_sensation[index, which_victim, :])):
                    which_victim = victim
            next_sensation = raw_sensation[index, which_victim, :]
            self.CanSeeIt = True

        elif not all(np.isnan(net_adj_mat[index, :])):
            temp_sensation = next_sensation.copy()
            num_scouts = np.sum(adj_mat[index, :])
            for ns in range(int(num_scouts)):
                curr_scout = np.argwhere(adj_mat[index, :])[ns]
                if any(sensation_evaluate[curr_scout, :][0].tolist()):
                    which_victim = np.argwhere(sensation_evaluate[curr_scout, :][0])[0]
                    for victim in np.argwhere(sensation_evaluate[curr_scout, :][0]):
                        if (np.linalg.norm(raw_sensation[curr_scout, victim, :]) <
                            np.linalg.norm(raw_sensation[curr_scout, which_victim, :])):
                            which_victim = victim

                    next_sensation[0] = (pos2pos[curr_scout, index][0][0] +
                                         raw_sensation[curr_scout, which_victim, :][0][0])
                    next_sensation[1] = (pos2pos[curr_scout, index][0][1] +
                                         raw_sensation[curr_scout, which_victim, :][0][1])
                    self.CanSeeIt = True

                    if np.linalg.norm(temp_sensation) < np.linalg.norm(next_sensation):
                        next_sensation = temp_sensation.copy()
                    else:
                        temp_sensation = next_sensation.copy()

        return next_sensation

    def sensation2index(self, sensation, max_vfd):
        if self.CanSeeIt:
            index = ((sensation[0] + max_vfd) * (2 * max_vfd + 1) + (sensation[1] + max_vfd))
        else:
            index = (2 * max_vfd + 1) ** 2

        return int(index)

    def rescue_accomplished(self, rescue_team_Hist, agent, adj_mat):
        if (((self.old_Sensation[0] == 0 and self.old_Sensation[1] == 0) or
             (self.curr_Sensation[0] == 0 and self.curr_Sensation[1] == 0)) and
                'r' in self.Role):
            self.Finish = True
            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 0)
            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 1)

            rescue_team_Hist.remove(agent)

        return rescue_team_Hist, adj_mat

    def victim_rescued(self, rescue_team_old_pos_list, rescue_team_curr_pos_list,
                       rescue_team_role_list, victim, victims_Hist):
        for idx, rescuer_old_pos in enumerate(rescue_team_old_pos_list):
            if (((rescuer_old_pos[0] == self.old_Pos[0] and
                  rescuer_old_pos[1] == self.old_Pos[1]) or
                 (rescue_team_curr_pos_list[idx][0] == self.old_Pos[0] and
                  rescue_team_curr_pos_list[idx][1] == self.old_Pos[1])) and
                    'r' in rescue_team_role_list[idx]):
                self.Finish = True
                victims_Hist.remove(victim)
                break  # You already removed this victim, no need to check the rest of the list

        return victims_Hist

    def convergence_check(self, accuracy):

        if (np.abs(np.sum(self.Q - self.Q_hist) /
                   (np.shape(self.Q)[0] * np.shape(self.Q)[1])) <= accuracy):
            self.Convergence = True

        return self.Convergence
