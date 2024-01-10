import numpy as np


class Robot:
    def __init__(self, id, pos, speed, algorithms, num_clusters, competency, capabilities, cap_list):
        self.id = id
        self.init_pos = pos
        self.pos = pos
        self.speed = speed
        self.w_rc = np.zeros((num_clusters,))
        self.algorithms = algorithms
        self.competency = competency
        self.capabilities = capabilities
        self.vectorized_cap = [1 if cap in self.capabilities else 0 for cap in cap_list]
        self.num_sensors = len(capabilities)

        self.tasks = []  # Assignment based on the capabilities
        self.tasks_full = []  # Assignment based on full satisfaction of the victims requirements

        self.tasks_init = []  # Assignment based on the number of tasks in the cluster (B2)
        self.tasks_init_dist = []  # Assigned task distance to the cluster center

        self.tasks_final = []  # Final Assignment based on the robots busy time (B3)
        self.tasks_finalized = []  # (B4)

        self.make_span = []
        self.abort_time = 0.0
        self.travel_time = 0.0

    def travel_cost(self):
        X_i = np.linalg.norm(np.subtract(self.tasks_init_dist[0], self.init_pos))
        X_f = np.linalg.norm(np.subtract(self.tasks_init_dist[-1], self.tasks_init_dist[0]))
        T_i = X_i / self.speed
        T_f = X_f / self.speed
        self.travel_time = T_f - T_i

    def reset(self):

        self.tasks = []  # Assignment based on the capabilities
        self.tasks_full = []  # Assignment based on full satisfaction of the victims requirements

        self.tasks_init = []  # Assignment based on the number of tasks in the cluster
        self.tasks_init_dist = []  # Assigned task distance to the cluster center

        self.tasks_final = []  # Final Assignment based on the robots busy time

        self.make_span = []
        self.abort_time = 0.0
        self.travel_time = 0.0

        self.w_rc = np.zeros_like(self.w_rc)


    def cap_list(self):
        self.capabilities
