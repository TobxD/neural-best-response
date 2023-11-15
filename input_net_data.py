import os

import yaml
from neural_policies import create_policy_net
import torch


def read_nn(game, path, config):
    policy_net = create_policy_net(game, config)
    policy_net.load_state_dict(torch.load(path))
    return policy_net


def read_all_nn(game, folder_path):
    nn_config = yaml.safe_load(open(os.path.join(folder_path, "mlp_config.yaml"), "r"))
    networks = []
    for file in os.listdir(folder_path):
        if file.endswith(".pkl"):
            networks.append(read_nn(game, os.path.join(folder_path, file), nn_config))
    return networks
