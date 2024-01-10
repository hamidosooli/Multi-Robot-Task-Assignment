import numpy as np


def ReqmentAnalysis(robot_list, task_list, robot_reqs, task_reqs):
    num_robots = len(robot_list)
    num_tasks = len(task_list)

    robot_and_task = task_reqs @ np.transpose(robot_reqs)
    task_reqs_sum = np.tile(np.sum(task_reqs, axis=1).reshape(num_tasks, 1), (1, num_robots))

    perfect_fulfill = np.logical_and(np.zeros_like(robot_and_task) < robot_and_task, robot_and_task == task_reqs_sum)
    perfect_fulfill_x, perfect_fulfill_y = np.where(np.transpose(perfect_fulfill))
    perfect_fulfill_list = [[b for a, b in zip(perfect_fulfill_x, perfect_fulfill_y) if a == i] for i in
                            range(max(perfect_fulfill_x)+1)]

    partial_fulfill = np.logical_and(np.zeros_like(robot_and_task) < robot_and_task, robot_and_task < task_reqs_sum)
    partial_fulfill_x, partial_fulfill_y = np.where(np.transpose(partial_fulfill))
    partial_fulfill_list = [[b for a, b in zip(partial_fulfill_x, partial_fulfill_y) if a == i] for i in
                            range(max(partial_fulfill_x) + 1)]

    [setattr(obj, attribute, value) for obj, value_1, value_2 in zip(robot_list, perfect_fulfill_list, partial_fulfill_list)
     for attribute, value in [("tasks_full", value_1), ("tasks", value_2)]]

    overall_fulfillment = np.logical_or(perfect_fulfill, partial_fulfill)
    who_helps_who = np.argwhere(overall_fulfillment)
    for candidate in who_helps_who:
        task_id = candidate[0]
        robot_id = candidate[1]
        check_req = np.logical_and(task_reqs[task_id, :], robot_reqs[robot_id, :])
        what_reqs = np.where(check_req)[0].tolist()

        for what_req in what_reqs:
            task_list[task_id].vectorized_candidates[what_req].append(robot_id)
    for task_id in range(num_tasks):
        for candidates_id in np.argwhere(task_reqs[task_id, :]):
            task_list[task_id].candidates.append(task_list[task_id].vectorized_candidates[candidates_id[0]])

def MissingCap(task_list, robot_reqs):
    unavaialable_caps = np.logical_not(np.logical_or.reduce(robot_reqs, 0))
    unavaialable_caps_ids = np.argwhere(unavaialable_caps)[0]
    missing_cap = []
    for task in task_list:
        for id in unavaialable_caps_ids:
            if task.vectorized_cap[id]:
                task.vectorized_rescued[id] = True
                missing_cap.append({task.id: id})
    print(missing_cap)
    return missing_cap
