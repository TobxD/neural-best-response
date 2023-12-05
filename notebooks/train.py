import sys
sys.path.append('..')
import importlib
import neural_policies
import main
import game_utils
import policy_gradient
from policy_gradient import PolicyGradientHypernetTrainer

import torch
import yaml
import numpy as np
import pyspiel
from types import SimpleNamespace
import wandb
import argparse

def do_work(args, game, hypernet_config, hypernet_train_config, nn_config):
    mlp = neural_policies.create_policy_net(game, nn_config)
    # hypernet = neural_policies.create_hypernet_nn_output(game, mlp, hypernet_config)
    hypernet = neural_policies.create_hypernet_actionoutput(game, mlp, hypernet_config)
    trainer = PolicyGradientHypernetTrainer(game, hypernet, hypernet_train_config, mlp)

    # trainer.train_best_response(nn_config, 1 - nn_player)
    trainer.train_best_response_q_baseline(nn_config, 1 - args.opponent_player)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn-config", type=str, default="../config/kuhn_mlp.yaml")
    parser.add_argument("--hypernet-config", type=str, default="../config/hypernet.yaml")
    parser.add_argument(
        "--hypernet-train-config",
        type=str,
        default="../config/hypernet_reinforce_train.yaml",
    )
    # only those games have been tested so far
    parser.add_argument(
        "--game",
        type=str,
        choices=["random", "kuhn_poker", "leduc_poker", "rps"],
        default="kuhn_poker",
    )
    parser.add_argument(
        "--opponent-player",
        type=int,
        choices=[0, 1],
        default=0,
    )
    # this is only for random games (NF)
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    game = main.create_game(args)
    nn_config = yaml.safe_load(open(args.nn_config, "r"))
    hypernet_config = yaml.safe_load(open(args.hypernet_config, "r"))
    hypernet_train_config = yaml.safe_load(open(args.hypernet_train_config, "r"))

    wandb.init(
        project="Game Solving Neural Best Response",
        config={
            "train_config": hypernet_train_config,
            "nn_config": nn_config,
            "hypernet_config": hypernet_config,
            "game": args.game,
            "opponent_player": args.opponent_player,
        }
    )

    do_work(args, game, hypernet_config, hypernet_train_config, nn_config)
