import argparse
import json
import os
import numpy as np
import torch
import yaml
from neural_policies import create_policy_net
import game_utils
from main import create_game
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from mlp import MLP


def sample_strategy(game, player):
    # TODO tune alpha, maybe 1/n
    alpha = 1 / 2

    states = game_utils.all_game_states(game, player)
    info_sets = set()
    strategy = []
    for state in states:
        if state.information_state_string(player) in info_sets:
            continue
        info_sets.add(state.information_state_string(player))
        action_mask = np.array(state.legal_actions_mask(player), dtype=bool)
        probabilities = np.random.dirichlet(alpha * np.ones(action_mask.sum()))
        action_probs = np.zeros(action_mask.shape)
        action_probs[action_mask] = probabilities
        strategy.append((state.information_state_tensor(player), action_probs))
    return strategy


def train_random_nn_policy(game, player, network, train_params):
    strategy = sample_strategy(game, player)
    inputs = torch.tensor([input_vector for input_vector, _ in strategy])
    outputs = torch.tensor([output_vector for _, output_vector in strategy])

    optimizer = optim.Adam(
        network.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
    )
    for step in range(train_params["num_steps"]):
        optimizer.zero_grad()
        ind = step % len(inputs)
        prediction = network(inputs)
        loss = nn.functional.cross_entropy(prediction, outputs)
        loss.backward()
        optimizer.step()
        # if step % 100 == 0:
        #     print(f"Step {step}, Loss: {loss.item()}")
        #     print(f"{(nn.functional.softmax(prediction, dim=-1)- outputs).abs().sum()}")
        #     print(f"{nn.functional.softmax(prediction, dim=-1)}\n{outputs}")
    return network


def train_networks(game, player, num_networks, mlp_config, train_config, output_folder):
    res = []
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "mlp_config.yaml"), "w") as f:
        yaml.dump(mlp_config, f)
    with open(os.path.join(output_folder, "train_config.yaml"), "w") as f:
        yaml.dump(train_config, f)
    num_pickles = len(
        [name for name in os.listdir(output_folder) if name.endswith(".pkl")]
    )

    for i in tqdm(range(num_networks)):
        network = create_policy_net(game, mlp_config)
        train_random_nn_policy(game, player, network, train_config)
        torch.save(
            network.state_dict(), os.path.join(output_folder, f"{num_pickles + i}.pkl")
        )
        res.append(network)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train networks for a game.")
    parser.add_argument("--game", type=str, required=True, help="Name of the game.")
    parser.add_argument(
        "--mlp-config", type=str, required=True, help="Path to MLP configuration file."
    )
    parser.add_argument(
        "--num-networks", type=int, required=True, help="Number of networks to train."
    )
    parser.add_argument("--player", type=int, required=True, help="Player number.")
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)

    args = parser.parse_args()

    mlp_config = yaml.safe_load(open(args.mlp_config, "r"))
    train_config = yaml.safe_load(open(args.train_config, "r"))

    game = create_game(args)
    trained_networks = train_networks(
        game,
        args.player,
        args.num_networks,
        mlp_config,
        train_config,
        args.output_folder,
    )
