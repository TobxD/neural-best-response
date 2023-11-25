import matplotlib.pyplot as plt

def extract_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    episode_values = {}
    current_episode = None
    for line in lines:
        if line.startswith('episode'):
            current_episode = int(line.split()[1])
            episode_values[current_episode] = []
        if line.startswith('value of table br policy') or line.startswith('value of hypernet br policy'):
            values = line.split('[')[1].split(']')[0].split()
            value_tuple = (float(values[0]), float(values[1]))
            episode_values[current_episode].append(value_tuple)

    return episode_values

def plot_traj_values(filename):
    values = extract_values(filename)
    episodes = list(values.keys())
    value_1 = [v[0][0] for v in values.values()]  #
    value_2 = [v[1][0] for v in values.values()]  #
    value_3 = [v[2][0] for v in values.values()]  #

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, value_1, label='Value of Table BR Policy')
    plt.plot(episodes, value_2, label='Value of Hypernet BR Policy (1st)')
    plt.plot(episodes, value_3, label='Value of Hypernet BR Policy (2nd)')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Values vs. Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    file_time = '20231125_132152'
    plot_traj_values(f'trajectory/evaluation_log_{file_time}.txt')