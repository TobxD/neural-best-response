import numpy as np
import pyspiel
import yaml
from policy_gradient import PolicyGradientTrainer, PolicyGradientHypernetTrainer
import neural_policies
import torch

from game_utils import (
    create_random_normal_game,
    nash_equilibrium_policy_and_value,
    compute_best_response_tabular_policy,
    combine_tabular_policies,
    policy_value,
)


def tabular_br(game, nn_player, nn_tab_policy):
    br = compute_best_response_tabular_policy(game, nn_tab_policy, 1 - nn_player)
    # print("br policy", br.action_probability_array)
    policies = [nn_tab_policy, br] if nn_player == 0 else [br, nn_tab_policy]
    total_policy = combine_tabular_policies(game, *policies)
    # print("total policy", total_policy.action_probability_array)
    print("nn policy with tabular br value", policy_value(game, total_policy))


def policy_gradient_br(
    game, nn_config, reinforce_config, nn_player, nn_policy, nn_tab_policy
):
    br_player = 1 - nn_player
    neural_br_policy = neural_policies.create_policy_net(game, nn_config)
    trainer = PolicyGradientTrainer(
        game, neural_br_policy, train_params=reinforce_config
    )
    trainer.train_best_response(nn_policy, br_player)

    neural_br_tab_policy = neural_policies.nn_to_tabular_policy(
        game, neural_br_policy, br_player
    )
    policies = (
        [nn_tab_policy, neural_br_tab_policy]
        if nn_player == 0
        else [
            neural_br_tab_policy,
            nn_tab_policy,
        ]
    )
    combined_policy = combine_tabular_policies(game, *policies)
    print("value of nn br policy", policy_value(game, combined_policy))


def train_hypernet_actionoutput(game, nn_config, nn_player, nn_policy):
    hypernet_config = yaml.safe_load(
        open("code/config/hypernet_actionoutput.yaml", "r")
    )
    train_config = yaml.safe_load(
        open("code/config/hypernet_reinforce_train.yaml", "r")
    )

    hypernet = neural_policies.create_hypernet_actionoutput(
        game, nn_policy, hypernet_config
    )
    trainer = PolicyGradientHypernetTrainer(game, hypernet, train_config)
    trainer.train_best_response(nn_config, 1 - nn_player)
    return hypernet


def train_hypernet_nn_output(game, nn_config, nn_player, nn_policy):
    """
    TODO we can optimize this code if we don't use the forward() but instead use the model_output() function
    """
    hypernet_config = yaml.safe_load(
        open("code/config/hypernet_actionoutput.yaml", "r")
    )
    train_config = yaml.safe_load(
        open("code/config/hypernet_reinforce_train.yaml", "r")
    )

    hypernet = neural_policies.create_hypernet_nn_output(
        game, nn_policy, hypernet_config, output_net=nn_policy
    )
    trainer = PolicyGradientHypernetTrainer(game, hypernet, train_config)
    trainer.train_best_response(nn_config, 1 - nn_player)
    return hypernet


def hypernet_actionoutput_br(game, hypernet, nn_player, nn_policy, nn_tab_policy):
    neural_br_tab_policy = neural_policies.nn_to_tabular_policy(
        game, hypernet, 1 - nn_player, input_net=nn_policy
    )
    policies = (
        [nn_tab_policy, neural_br_tab_policy]
        if nn_player == 0
        else [
            neural_br_tab_policy,
            nn_tab_policy,
        ]
    )
    combined_policy = combine_tabular_policies(game, *policies)
    print("value of hypernet br policy", policy_value(game, combined_policy))


def main(game, nn_config, reinforce_config):
    # print("nash", nash_equilibrium_policy_and_value(game))

    nn_player = 0
    nn_policy = neural_policies.create_policy_net(game, nn_config)
    nn_tab_policy = neural_policies.nn_to_tabular_policy(game, nn_policy, nn_player)
    print("nn tab policy", nn_tab_policy.action_probability_array)

    # hypernet = train_hypernet_actionoutput(game, nn_config, nn_player, nn_policy)
    hypernet_nn = train_hypernet_actionoutput(game, nn_config, nn_player, nn_policy)

    for i in range(20):
        print(f"==== eval game {i} ====")
        nn_policy = neural_policies.create_policy_net(game, nn_config)
        nn_tab_policy = neural_policies.nn_to_tabular_policy(game, nn_policy, nn_player)
        tabular_br(game, nn_player, nn_tab_policy)
        policy_gradient_br(
            game, nn_config, reinforce_config, nn_player, nn_policy, nn_tab_policy
        )
        hypernet_actionoutput_br(game, hypernet_nn, nn_player, nn_policy, nn_tab_policy)


def create_game(args):
    if args.game == "random":
        return create_random_normal_game(args.num_actions)
    else:
        return pyspiel.load_game(args.game)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nn-config", type=str, default="code/config/mlp.yaml")
    parser.add_argument(
        "--reinforce-config", type=str, default="code/config/reinforce_train.yaml"
    )
    # only those games have been tested so far
    parser.add_argument(
        "--game",
        type=str,
        choices=["random", "kuhn_poker", "leduc_poker"],
        default="random",
    )
    # this is only for random games (NF)
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    game = create_game(args)
    nn_config = yaml.safe_load(open(args.nn_config, "r"))
    reinforce_config = yaml.safe_load(open(args.reinforce_config, "r"))
    main(game, nn_config, reinforce_config)
