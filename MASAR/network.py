import numpy as np


class Network:
    def __init__(self, adj_mat, num_agents, num_victims):
        self.adj_mat = adj_mat.copy()
        self.adj_mat[self.adj_mat == 0] = np.nan
        self.num_agents = num_agents
        self.num_victims = num_victims

    def pos2pos(self, pos_list):

        pos_array = np.empty((self.num_agents, self.num_agents, 2))
        pos_array[:, :, 0] = np.tile(pos_list[:, 0].reshape(self.num_agents, 1), (1, self.num_agents))
        pos_array[:, :, 1] = np.tile(pos_list[:, 1].reshape(self.num_agents, 1), (1, self.num_agents))

        pos2pos = np.subtract(pos_array, np.transpose(pos_array, (1, 0, 2)))
        return pos2pos

    def sensed_pos(self, victim_pos_list, rs_pos_list):
        self.num_victims = len(victim_pos_list)

        rs_pos_array = np.empty((self.num_agents, self.num_victims, 2))
        rs_pos_array[:, :, 0] = np.tile(rs_pos_list[:, 0].reshape(self.num_agents, 1), (1, self.num_victims))
        rs_pos_array[:, :, 1] = np.tile(rs_pos_list[:, 1].reshape(self.num_agents, 1), (1, self.num_victims))

        victim_pos_array = np.empty((self.num_agents, self.num_victims, 2))

        victim_pos_array[:, :, 0] = np.tile(victim_pos_list[:, 0].reshape(1, self.num_victims), (self.num_agents, 1))
        victim_pos_array[:, :, 1] = np.tile(victim_pos_list[:, 1].reshape(1, self.num_victims), (self.num_agents, 1))

        return np.subtract(victim_pos_array, rs_pos_array)

    def is_seen(self, vfd_list, raw_sensation, vfd_status):
        vfd_mat = np.tile(vfd_list.reshape(self.num_agents, 1), (1, self.num_victims))
        in_vfd_condition = np.zeros_like(raw_sensation)
        in_vfd_condition[:, :, 0] = in_vfd_condition[:, :, 1] = vfd_mat
        tuple_cond = np.abs(raw_sensation) <= in_vfd_condition
        in_vfd = np.logical_and(tuple_cond[:, :, 0], tuple_cond[:, :, 1])

        idx_vfd_status = in_vfd_condition + raw_sensation

        vfd_status_condition = in_vfd.copy()
        for agent_id in range(len(vfd_list)):
            for victim_id, first_cond in enumerate(in_vfd[agent_id]):
                if first_cond:
                    if vfd_status[agent_id][int(idx_vfd_status[agent_id, victim_id][0]),
                                            int(idx_vfd_status[agent_id, victim_id][1])]:
                        vfd_status_condition[agent_id, victim_id] = True
        return vfd_status_condition
