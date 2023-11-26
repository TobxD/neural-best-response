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


def plot_traj_values(filename, base, time, name, title):
    values = extract_values(filename)
    episodes = list(values.keys())
    value_1 = [float(v[0]) for v in values.values()]  
    value_2 = [float(v[1]) for v in values.values()]  
    value_3 = [float(v[2]) for v in values.values()]

    plt.figure(figsize=(10, 6))
    # plt.plot(episodes, value_1, label='Expected Value')
    plt.plot(episodes, value_2, label='Value of Hypernet BR Policy (softmax temp=1)', c = 'orange')
    plt.plot(episodes, value_3, label='Value of Hypernet BR Policy (softmax temp=0)', c = 'green')
    plt.axhline(y=value_1[0], linestyle='--', c = 'grey', label='theoretical maximum')
    plt.axhline(base, linestyle='--', c='red', label='fixed network baseline')  
    plt.ylim(0, 0.7)
    plt.xlim(0, 100000)
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.savefig(f'trajectory/20231126_{time}/{name}.png', dpi=300)
    plt.close()

if __name__=='__main__':
    base_train_p0 = 0.42634892660048623
    base_eval_p0 = 0.4223601409249119
    base_train_p1 = 0.510115206566238
    base_eval_p1 = 0.47288730269034057
    date = '20231126'
    time = '1127'
    plot_traj_values(f'trajectory/20231126_{time}/train_evaluation_log.txt', base_train_p0, time, '0_train', 'training trajectory for player 0')
    plot_traj_values(f'trajectory/20231126_{time}/eval_evaluation_log.txt', base_eval_p0, time, '0_eval', 'testing trajectory for player 0')

    time = '1206'
    plot_traj_values(f'trajectory/20231126_{time}/train_evaluation_log.txt', base_train_p1,time,'1_train', 'training trajectory for player 1')
    plot_traj_values(f'trajectory/20231126_{time}/eval_evaluation_log.txt', base_eval_p1,time,'1_eval', 'testing trajectory for player 1')

    time = '1218'
    plot_traj_values(f'trajectory/20231126_{time}/train_evaluation_log.txt', base_train_p0,time,'0_train', 'training trajectory for player 0')
    plot_traj_values(f'trajectory/20231126_{time}/eval_evaluation_log.txt', base_eval_p0,time,'0_eval', 'testing trajectory for player 0')

    time = '1220'
    plot_traj_values(f'trajectory/20231126_{time}/train_evaluation_log.txt', base_train_p1,time,'1_train', 'training trajectory for player 1')
    plot_traj_values(f'trajectory/20231126_{time}/eval_evaluation_log.txt', base_eval_p1,time,'1_eval', 'testing trajectory for player 1')






    """
    =============================
    # 1127:
    nn_player 0
    [128,128,128,128,128,128,128]
    states + weights
    =============================
    # 1206:
    nn_player 1
    [128,128,128,128,128,128,128]
    states + weights
    =============================

    
    ---------------------TODO----------------------------
    # 1218:
    nn_player 0
    [128,128,128,128,128,128,128]
    weights

    # 1220:
    nn_player 1
    [128,128,128,128,128,128,128]
    weights

    """
    
