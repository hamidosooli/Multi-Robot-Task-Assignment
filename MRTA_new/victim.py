import numpy as np


class Victim:
    def __init__(self, id, pos, make_span, requirements, cap_list):
        self.id = id
        self.pos = pos
        self.capabilities = requirements  # Victim requirements
        for cap_id, cap in enumerate(self.capabilities):
            if isinstance(cap, bytes):
                self.capabilities[cap_id] = cap.decode()
        self.vectorized_cap = [1 if cap in self.capabilities else 0 for cap in cap_list]
        self.rem_req = []  # Victim remaining requirements
        self.vectorized_candidates = [[] for _ in range(len(self.vectorized_cap))]
        self.candidates = []
        self.cluster_id = np.nan
        self.cluster_dist = np.nan
        self.cluster_dist_NW = np.nan
        self.rescued = np.zeros_like(requirements, dtype=bool)
        self.vectorized_rescued = np.zeros_like(self.vectorized_cap, dtype=bool)
        self.make_span = make_span
        self.health_stt = np.random.choice([1, .6, .3])  # 1: high health, .6: low health, .3:critical health
