import matplotlib.pyplot as plt
import numpy as np

def extract_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    episode_values = {}
    current_episode = None
    for line in lines:
        if line.startswith('episode'):
            current_episode = int(line.split()[1])
            episode_values[current_episode] = []
        else:
            values = line.split()[3]
            episode_values[current_episode].append(values)

    return episode_values

def smooth_list(values, window_size):
    smoothed_values = np.convolve(np.array(values), np.ones(window_size)/window_size, mode='valid')
    return smoothed_values


def plot_traj_values(filename):
    values = extract_values(filename)
    episodes = list(values.keys())
    value_1 = [float(v[0]) for v in values.values()]  
    value_2 = [float(v[1]) for v in values.values()]  
    value_3 = [float(v[2]) for v in values.values()]





    plt.figure(figsize=(10, 6))
    # plt.plot(episodes, value_1, label='Value of Table BR Policy')
    plt.plot(episodes, value_2, label='Value of Hypernet BR Policy (1st)')
    plt.plot(episodes, value_3, label='Value of Hypernet BR Policy (2nd)')
    plt.axhline(y=value_1[0], linestyle='--')  
    plt.ylim(0, 0.6)
    plt.yticks([i/10.0 for i in range(11)])
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Values vs. Episodes')
    plt.legend()
    # plt.grid(True)
    plt.show()

if __name__=='__main__':
    file_time = '20231126_090030'
    plot_traj_values(f'trajectory/evaluation_log_{file_time}.txt')
