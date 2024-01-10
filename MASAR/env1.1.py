import time

import numpy as np
import h5py
import json

from action_selection import eps_greedy
from network import Network
from agent import Agent

NUM_EPISODES = 500
NUM_RUNS = 100
Multi_Runs = False
# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
num_Acts = len(ACTIONS)

# Environment dimensions
Row_num = 20
Col_num = 20
row_lim = Row_num - 1
col_lim = Col_num - 1

#                          r1 r2
adj_mat_prior = np.array([[0, 0],
                          [0, 0]], dtype=float)
exp_name = 'FindSurvivors'

# make the map from json file
# with open('data10.json') as f:
#     data = json.load(f)
#     test = data['map'][0]
#     dim = data['dimensions']
#     rows = dim[0]['rows']
#     columns = dim[0]['columns']
#
#     env_map = np.zeros((rows, columns))
#
#     for cell in data['map']:
#         if cell['isWall'] == 'true':
#             env_map[cell['x'], cell['y']] = 1

env_mat = np.zeros((Row_num, Col_num))
global env_map
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
# env_map=np.zeros((20, 20))
# Transition function (avoid walls)
def movement(pos, action, speed):
    global env_map
    row = pos[0]
    col = pos[1]
    next_pos = pos.copy()
    if action == 0:  # up
        next_pos = [max(row - speed, 0), col]
    elif action == 1:  # down
        next_pos = [min(row + speed, row_lim), col]
    elif action == 2:  # right
        next_pos = [row, min(col + speed, col_lim)]
    elif action == 3:  # left
        next_pos = [row, max(col - speed, 0)]
    if env_map[next_pos[0], next_pos[1]] == 0:
        return next_pos
    else:
        return pos


def reward_func(sensation_prime):
    if sensation_prime[0] == 0 and sensation_prime[1] == 0:
        re = 1
    else:
        re = -.1
    return re


def q_learning(q, old_idx, curr_idx, re, act, alpha=0.8, gamma=0.9):
    q[old_idx, act] += alpha * (re + gamma * np.max(q[curr_idx, :]) - q[old_idx, act])
    return q


def env(accuracy=1e-15):
    global adj_mat_prior
    # Define the Network and the agent objects
    network = Network
    agent = Agent

    # Define the rescue team
    r1 = agent(0, 'r', 5, 5, 1, np.argwhere(env_map == 0)[np.random.randint(len(np.argwhere(env_map == 0)))],
               num_Acts, Row_num, Col_num)
    r2 = agent(1, 'r', 3, 3, 1, np.argwhere(env_map == 0)[np.random.randint(len(np.argwhere(env_map == 0)))],
               num_Acts, Row_num, Col_num)
    # s3 = agent(2, 's', 4, 4, 1, [row_lim, 0], num_Acts, Row_num, Col_num)
    # s4 = agent(3, 's', 4, 4, 1, [0, col_lim], num_Acts, Row_num, Col_num)
    # rs5 = agent(4, 'r', 4, Row_num, 1, [row_lim, col_lim], num_Acts, Row_num, Col_num)

    # Define the victims
    v1 = agent(0, 'v', 0, 0, 1, np.argwhere(env_map == 0)[np.random.randint(len(np.argwhere(env_map == 0)))],
               num_Acts, Row_num, Col_num)
    v2 = agent(1, 'v', 0, 0, 1, np.argwhere(env_map == 0)[np.random.randint(len(np.argwhere(env_map == 0)))],
               num_Acts, Row_num, Col_num)
    # v3 = agent(2, 'v', 0, 0, 1, [int(Row_num / 2) - 2, int(Col_num / 2) - 2], num_Acts, Row_num, Col_num)
    # v4 = agent(3, 'v', 0, 0, 1, [int(Row_num / 2) + 4, int(Col_num / 2) + 4], num_Acts, Row_num, Col_num)
    # v5 = agent(4, 'v', 0, 0, 1, [int(Row_num / 2) - 4, int(Col_num / 2) - 4], num_Acts, Row_num, Col_num)

    # List of objects
    rescue_team = [r1, r2]
    victims = [v1, v2]
    VFD_list = []

    num_just_scouts = 0
    rescue_team_roles = []


    for agent in rescue_team:
        rescue_team_roles.append(agent.Role)
        # List of the Visual Fields
        VFD_list.append(agent.VisualField)

        # Count the number of just scouts
        if agent.Role == 's':
            num_just_scouts += 1
    # eps = -1
    tic = time.time()
    # while True:
    for eps in range(NUM_EPISODES):
        rescue_team_Hist = rescue_team.copy()
        victims_Hist = victims.copy()
        adj_mat = adj_mat_prior.copy()

        agents_idx = []
        for agent in rescue_team:
            agents_idx.append(agent.id)

        victims_idx = []
        for victim in victims:
            victims_idx.append(victim.id)
        rescue_team_roles = np.array(rescue_team_roles, dtype=list)
        # eps += 1

        # Reset the agents flags, positions, etc
        for agent in rescue_team:
            agent.reset()
        # Reset the victims flags, positions, etc
        for victim in victims:
            victim.reset()

        t_step = 0
        # for _ in range(100):
        while True:
            num_rescue_team = len(rescue_team_Hist)
            num_victims = len(victims_Hist)

            net = network(adj_mat, num_rescue_team, num_victims)

            t_step += 1

            rescue_team_VFD_list = []
            team_VFD_status = []
            for agent in rescue_team_Hist:
                # List of the Visual Fields
                rescue_team_VFD_list.append(agent.VisualField)

                # Count the steps that agent could see a victim
                if agent.CanSeeIt:
                    agent.t_step_seen += 1

                # Keeping track of the rescue team positions
                agent.Traj.append(agent.old_Pos)

                # Update VFD status
                agent.update_vfd(env_map)

                # Keep track of VFD status
                agent.VFD_status_history.append(agent.vfd_status)

                # VFD status for the team
                team_VFD_status.append(agent.vfd_status)

                # History of Q
                agent.Q_hist = agent.Q.copy()

            rescue_team_VFD_list = np.asarray(rescue_team_VFD_list)

            # Keep track of the victims positions
            # Make a list of the victims old positions
            victims_old_pos_list = []
            for victim in victims_Hist:
                victim.Traj.append(victim.old_Pos)
                victims_old_pos_list.append(victim.old_Pos)
            victims_old_pos_list = np.asarray(victims_old_pos_list)

            # Make a list of the agents old positions
            rescue_team_old_pos_list = []
            for agent in rescue_team_Hist:
                rescue_team_old_pos_list.append(agent.old_Pos)
            rescue_team_old_pos_list = np.asarray(rescue_team_old_pos_list)

            # Calculation of the distance between the agents
            old_scouts2rescuers = net.pos2pos(rescue_team_old_pos_list)

            # Calculation of the raw sensations for the rescue team
            old_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_old_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_old_sensations = net.is_seen(rescue_team_VFD_list, old_raw_sensations, team_VFD_status)

            rescue_team_curr_pos_list = []
            rescue_team_role_list = []

            for agent in rescue_team_Hist:
                # Calculation of the sensations for the rescue team
                agent.old_Sensation = agent.update_sensation(rescue_team_Hist.index(agent),
                                                             old_raw_sensations, eval_old_sensations,
                                                             old_scouts2rescuers, net.adj_mat, adj_mat)
                # Calculation of the indices for the rescue team
                agent.old_Index = agent.sensation2index(agent.old_Sensation, agent.max_VisualField)

                # Actions for the rescue team
                agent.action = eps_greedy(agent.Q[agent.old_Index, :], ACTIONS)

                # Next positions for the rescue team
                agent.curr_Pos = movement(agent.old_Pos, agent.action, agent.Speed)

                # Search algorithm
                # agent.straight_move(agent.old_Index, agent.wereHere, env_map)
                # agent.random_walk(agent.old_Index, agent.old_Pos, agent.Speed, env_map)
                agent.ant_colony_move(env_mat, agent.old_Index, env_map)
                # agent.levy_walk(env_map)
                # List of the current positions for the rescue team members
                rescue_team_curr_pos_list.append(agent.curr_Pos)

                # List of the roles for the rescue team members
                rescue_team_role_list.append(agent.Role)

            rescue_team_curr_pos_list = np.asarray(rescue_team_curr_pos_list)

            # Calculation of the distance between agents (after their movement)
            curr_scouts2rescuers = net.pos2pos(rescue_team_curr_pos_list)

            # Calculation of the new raw sensations for the rescue team (after their movement)
            curr_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_curr_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_curr_sensations = net.is_seen(rescue_team_VFD_list, curr_raw_sensations, team_VFD_status)

            # Calculation of the new sensations for the rescue team (after their movement)
            for agent in rescue_team_Hist:
                agent.curr_Sensation = agent.update_sensation(rescue_team_Hist.index(agent),
                                                              curr_raw_sensations, eval_curr_sensations,
                                                              curr_scouts2rescuers, net.adj_mat, adj_mat)
                # Calculation of the indices for the rescue team (after their movement)
                agent.curr_Index = agent.sensation2index(agent.curr_Sensation, agent.max_VisualField)

                # Rewarding the rescue team
                agent.reward = reward_func(agent.curr_Sensation)

                # Q learning for the rescue team
                agent.Q = q_learning(agent.Q, agent.old_Index, agent.curr_Index, agent.reward, agent.action, alpha=0.8)

                # Check to see if the team rescued any victim
                if not agent.Finish:
                    rescue_team_Hist, adj_mat = agent.rescue_accomplished(rescue_team_Hist, agent, adj_mat)
                    # Keeping track of the rewards
                    agent.RewHist.append(agent.reward)
                    if agent.CanSeeIt:
                        agent.RewHist_seen.append(agent.reward)
                    if agent.Finish and agent.First:
                        agent.Steps.append(t_step)
                        agent.Steps_seen.append(agent.t_step_seen)
                        agent.RewSum.append(np.sum(agent.RewHist))
                        agent.RewSum_seen.append(np.sum(agent.RewHist_seen))
                        rescue_team[agent.id] = agent
                        agent.First = False
                        for victim in victims_Hist:
                            # Check to see if the victim rescued by the team
                            # Keep track of the steps
                            # Remove the victim from the list
                            if not victim.Finish:
                                victims[victim.id] = victim
                                victims_Hist = victim.victim_rescued(rescue_team_old_pos_list,
                                                                     rescue_team_curr_pos_list,
                                                                     rescue_team_role_list,
                                                                     victim, victims_Hist)
                                if victim.Finish and victim.First:
                                    victim.Steps.append(t_step)
                                    victim.First = False
                                    break  # Rescue more than one victim by an agent
            if len(rescue_team_Hist) == num_just_scouts and len(victims_Hist) == 0:
                print(f'In episode {eps+1}, all of the victims were rescued in {t_step} steps')
                break

            # Update the rescue team positions
            for agent in rescue_team_Hist:
                agent.old_Pos = agent.curr_Pos

            # Victims' actions and positions
            for victim in victims_Hist:
                # Actions for the victims
                victim.action = np.random.choice(ACTIONS)
                # Victims next positions
                victim.curr_Pos = movement(victim.old_Pos, victim.action, victim.Speed)
                # Update the victims position
                victim.old_Pos = victim.curr_Pos

        # Check for the proper number of episodes
        # convergence_flag = []
        # for agent in rescue_team:
        #     convergence_flag.append(agent.convergence_check(accuracy))
        # if all(convergence_flag):
        #     break

    # Add agents last pos in the trajectory
    for agent in rescue_team:
        for victim in victims:
            if agent.curr_Pos[0] == victim.old_Pos[0] and agent.curr_Pos[1] == victim.old_Pos[1]:
                agent.Traj.append(agent.curr_Pos)
                agent.VFD_status_history.append(agent.vfd_status)

    rescue_team_Traj = []
    VFD_status_list = []
    rescue_team_RewSum = []
    rescue_team_Steps = []
    rescue_team_RewSum_seen = []
    rescue_team_Steps_seen = []
    rescue_team_Q = []
    largest = len(rescue_team[0].Traj)
    for agent in rescue_team:
        if len(agent.Traj) > largest:
            largest = len(agent.Traj)
        rescue_team_RewSum.append(agent.RewSum)
        rescue_team_Steps.append(agent.Steps)
        rescue_team_RewSum_seen.append(agent.RewSum_seen)
        rescue_team_Steps_seen.append(agent.Steps_seen)
        rescue_team_Q.append(agent.Q)
    for agent in rescue_team:
        while len(agent.Traj) < largest:
            agent.Traj.append(agent.Traj[-1])
            agent.VFD_status_history.append((agent.vfd_status))
        rescue_team_Traj.append(agent.Traj)
        # List of the VFD status
        VFD_status_list.append(agent.VFD_status_history)

    victims_Traj = []
    for victim in victims:
        while len(victim.Traj) < largest:
            victim.Traj.append(victim.Traj[-1])
        victims_Traj.append(victim.Traj)
    print(f'This experiment took {time.time() - tic} seconds')
    return (rescue_team_Traj,
            rescue_team_RewSum, rescue_team_Steps,
            rescue_team_RewSum_seen, rescue_team_Steps_seen,
            rescue_team_Q, victims_Traj, VFD_list, VFD_status_list, rescue_team_roles)


if Multi_Runs:
    # Multi Runs
    rescue_team_RewSum_Run = []
    rescue_team_Steps_Run = []
    rescue_team_RewSum_seen_Run = []
    rescue_team_Steps_seen_Run = []
    for run in range(NUM_RUNS):
        print(f'Run {run + 1} of {NUM_RUNS}')
        (rescue_team_Traj,
         rescue_team_RewSum, rescue_team_Steps,
         rescue_team_RewSum_seen, rescue_team_Steps_seen,
         rescue_team_Q, victims_Traj, VFD_list, VFD_status_list, rescue_team_roles) = env(accuracy=1e-7)

        rescue_team_RewSum_Run.append(list(filter(None, rescue_team_RewSum)))
        rescue_team_Steps_Run.append(list(filter(None, rescue_team_Steps)))
        rescue_team_RewSum_seen_Run.append(list(filter(None, rescue_team_RewSum_seen)))
        rescue_team_Steps_seen_Run.append(list(filter(None, rescue_team_Steps_seen)))

    rescue_team_RewSum_Run = np.mean(np.asarray(rescue_team_RewSum_Run), axis=0)
    rescue_team_Steps_Run = np.mean(np.asarray(rescue_team_Steps_Run), axis=0)
    rescue_team_RewSum_seen_Run = np.mean(np.asarray(rescue_team_RewSum_seen_Run), axis=0)
    rescue_team_Steps_seen_Run = np.mean(np.asarray(rescue_team_Steps_seen_Run), axis=0)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}_{str(NUM_RUNS)}Runs.hdf5', 'w') as f:
        for idx, rew_sum in enumerate(rescue_team_RewSum_Run):
            f.create_dataset(f'RS{idx}_reward', data=rew_sum)
        for idx, steps in enumerate(rescue_team_Steps_Run):
            f.create_dataset(f'RS{idx}_steps', data=steps)
        for idx, rew_sum_seen in enumerate(rescue_team_RewSum_seen_Run):
            f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
        for idx, steps_seen in enumerate(rescue_team_Steps_seen_Run):
            f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
        f.create_dataset('RS_VFD', data=VFD_list)

else:
    # Single Run
    (rescue_team_Traj,
     rescue_team_RewSum, rescue_team_Steps,
     rescue_team_RewSum_seen, rescue_team_Steps_seen,
     rescue_team_Q, victims_Traj, VFD_list, VFD_status_list, rescue_team_roles) = env(accuracy=1e-7)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}.hdf5', 'w') as f:
        for idx, traj in enumerate(rescue_team_Traj):
            f.create_dataset(f'RS{idx}_trajectory', data=traj)
        for idx, vfd_sts in enumerate(VFD_status_list):
            f.create_dataset(f'RS{idx}_VFD_status', data=vfd_sts)
        for idx, rew_sum in enumerate(rescue_team_RewSum):
            f.create_dataset(f'RS{idx}_reward', data=rew_sum)
        for idx, steps in enumerate(rescue_team_Steps):
            f.create_dataset(f'RS{idx}_steps', data=steps)
        for idx, rew_sum_seen in enumerate(rescue_team_RewSum_seen):
            f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
        for idx, steps_seen in enumerate(rescue_team_Steps_seen):
            f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
        for idx, q in enumerate(rescue_team_Q):
            f.create_dataset(f'RS{idx}_Q', data=q)
        for idx, victim_traj in enumerate(victims_Traj):
            f.create_dataset(f'victim{idx}_trajectory', data=victim_traj)
        f.create_dataset('victims_num', data=[len(victims_Traj)])
        f.create_dataset('RS_VFD', data=VFD_list)
        f.create_dataset('RS_ROLES', data=rescue_team_roles)
