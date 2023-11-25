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

def get_traj_values(filename= 'trajectory/evaluation_log_20231125_132152.txt'):
    values = extract_values(filename)
    for episode, vals in values.items():
        print(f"Episode {episode}: {vals}")
    return values

if __name__=='__main__':
    get_traj_values()