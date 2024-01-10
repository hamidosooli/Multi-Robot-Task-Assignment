from gridworld_multi_agent1_1 import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py
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
exp_name = '4RS_1000episodes'
# exp_name = '2R_NS'
# exp_name = '2R_2S'
# exp_name = '2R_2S_A2A'
# exp_name = '2R_NS_1V'
# exp_name = '2R_2S_1V'
# exp_name = '2R_2S_A2A_1V'
# exp_name = '5R_5S'

plt.rcParams.update({'font.size': 22})
file_name = f'multi_agent_Q_learning_{exp_name}.hdf5'

run_animate = False

rescue_team_Traj = []
rescue_team_VFD_status = []
rescue_team_RewSum = []
rescue_team_Steps = []
rescue_team_RewSum_seen = []
rescue_team_Steps_seen = []
rescue_team_Q = []
victims_Traj = []

with h5py.File(file_name, 'r') as f:

    for idx in range(len(f['RS_VFD'])):

        rescue_team_Traj.append(f[f'RS{idx}_trajectory'])
        rescue_team_VFD_status.append(np.asarray(f[f'RS{idx}_VFD_status']))
        rescue_team_RewSum.append(f[f'RS{idx}_reward'])
        rescue_team_Steps.append(f[f'RS{idx}_steps'])
        rescue_team_RewSum_seen.append(f[f'RS{idx}_reward_seen'])
        rescue_team_Steps_seen.append(f[f'RS{idx}_steps_seen'])
        rescue_team_Q.append(f[f'RS{idx}_Q'])
    for idx in range(f['victims_num'][0]):
        victims_Traj.append(f[f'victim{idx}_trajectory'])

    if run_animate:
        animate(np.asarray(rescue_team_Traj), np.asarray(victims_Traj),
                np.asarray(f['RS_VFD']), rescue_team_VFD_status, f['RS_ROLES'], env_map, wait_time=.01)

    rescue_team_legends = []

    plt.figure('reward')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_RewSum[idx])[::10])
        rescue_team_legends.append(f'Agent {idx+1}')
    plt.xlabel('Number of episodes')
    plt.ylabel('Rescue Team Total Rewards')
    plt.legend(rescue_team_legends)

    plt.figure('reward_seen')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_RewSum_seen[idx]))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescue Team Rewards During Victim Visit')
    plt.legend(rescue_team_legends)

    plt.figure('steps')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_Steps[idx]))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescue Team Total Steps')
    plt.legend(rescue_team_legends)

    plt.figure('steps_seen')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_Steps_seen[idx]))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescue Team Steps During Victim Visit')
    plt.legend(rescue_team_legends)

    plt.show()
