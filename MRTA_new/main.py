import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from ReqmentAnalysis import ReqmentAnalysis, MissingCap
from Clustering import Clustering, ClstrAsgn
from maps import env_map2
from robot import Robot
from victim import Victim
from PerfAnalysis import VictimAssign, RobotAssign
import h5py


capabilities = ['FirstAids', 'DebrisRemover', 'OxygenCylinder', 'Defuser', 'Manipulator', 'FireExtinguisher']
make_span = {'FirstAids': 15, 'DebrisRemover': 30, 'OxygenCylinder': 20,
             'Defuser': 10, 'Manipulator': 45, 'FireExtinguisher': 35}
num_clusters = 4
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

plt.rcParams.update({'font.size': 22})

env_map = env_map2
num_rows, num_cols = np.shape(env_map)
ox, oy = np.where(env_map == 1)

victims = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]


r0 = Robot(0, [0, 9], 0.2, [], num_clusters, [], ['Defuser', 'DebrisRemover'], capabilities)
r1 = Robot(1, [0, 10],  0.3, [], num_clusters, [], ['FirstAids', 'OxygenCylinder', 'Manipulator'], capabilities)
r2 = Robot(2, [19, 9],  0.4, [], num_clusters, [], ['Manipulator'], capabilities)
r3 = Robot(3, [19, 10], 0.3, [], num_clusters, [], ['FirstAids', 'Defuser'], capabilities)

robots = [r0, r1, r2, r3]
starts = []
for robot in robots:
    starts.append(robot.pos)
victim_reqs = []
# Go through the list of tasks
for victim in victims:
    victim_reqs.append(victim.vectorized_cap)
victim_reqs = np.asarray(victim_reqs)

robot_reqs = []
# Go through the list of robots
for robot in robots:
    robot_reqs.append(robot.vectorized_cap)
robot_reqs = np.asarray(robot_reqs)
ReqmentAnalysis(robots, victims, robot_reqs, victim_reqs)
MissingCap(victims, robot_reqs)
print(f'Robot 0 tasks based on capabilities: {r0.tasks}\n'
      f'Robot 1 tasks based on capabilities: {r1.tasks}\n'
      f'Robot 2 tasks based on capabilities: {r2.tasks}\n'
      f'Robot 3 tasks based on capabilities: {r3.tasks}\n')

print(f'Robot 0 tasks based on full satisfaction of the capabilities: {r0.tasks_full}\n'
      f'Robot 1 tasks based on full satisfaction of the capabilities: {r1.tasks_full}\n'
      f'Robot 2 tasks based on full satisfaction of the capabilities: {r2.tasks_full}\n'
      f'Robot 3 tasks based on full satisfaction of the capabilities: {r3.tasks_full}\n')

clusters, clusters_coord = Clustering(num_clusters, victims, ox, oy)
victims_new = ClstrAsgn(robots, victims, clusters, clusters_coord, num_clusters)
travel2clusters = []
for robot in robots:
    travel2clusters.append(robot.pos)
victims_new = VictimAssign(robots, victims, victims_new)
victims_new = RobotAssign(robots, victims_new, victims, ox, oy)
for robot in robots:
    print(f'Robot{robot.id}-->B2:{robot.tasks_init}, B3:{robot.tasks_final}, B4:{robot.tasks_finalized}')

with h5py.File(f'MRTA.hdf5', 'w') as f:
    f.create_dataset(f'RS_size', data=len(robots))
    f.create_dataset(f'Victims_size', data=len(victims))
    f.create_dataset(f'RS_starts', data=starts)
    f.create_dataset(f'RS_travel2clusters', data=travel2clusters)
    for robot in robots:
        print(f'{robot.id} --> {robot.tasks_init} & {robot.tasks_final} & {robot.tasks_finalized}')

        f.create_dataset(f'RS{robot.id}_Step_1', data=robot.tasks_init)
        f.create_dataset(f'RS{robot.id}_Step_2', data=robot.tasks_final)
        f.create_dataset(f'RS{robot.id}_Step_3', data=robot.tasks_finalized)


fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.rcParams.update({'font.size': 50})
for idx, cluster in enumerate(clusters_coord):
    ax.scatter(cluster[0], cluster[1], c="red", marker="^")
    ax.text(cluster[0], cluster[1], f'C{idx}')

for victim in victims:
    # print(victim.rescued)
    ax.scatter(victim.pos[1]-1, victim.pos[0]+1, c="blue", marker="s")
    ax.text(victim.pos[1]-1, victim.pos[0]+1, f'V{victim.id}')
vor = Voronoi(clusters_coord)
for p_id, p in enumerate(vor.points):
    vor.points[p_id] = p[::-1]

for p_id, p in enumerate(vor.vertices):
    vor.vertices[p_id] = p[::-1]

for p_id, p in enumerate(vor.ridge_points):
    vor.ridge_points[p_id] = p[::-1]

for p_id, p in enumerate(vor.regions):
    vor.regions[p_id] = p[::-1]

voronoi_plot_2d(vor, ax)
plt.plot(dpi=1200)
plt.xticks([])
plt.yticks([])
ax.invert_yaxis()
plt.show()