import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from open_spiel.python import policy as policy_module

from game_utils import all_game_states


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for layer_dim in layers:
            linear_layer = nn.Linear(prev_dim, layer_dim)
            init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
            # we can't use just 0 bias if we have 0 input for NF games
            init.uniform_(linear_layer.bias)
            self.layers.append(linear_layer)
            prev_dim = layer_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        init.kaiming_normal_(self.output_layer.weight, nonlinearity="relu")
        # we can't use just 0 bias if we have 0 input for NF games
        init.uniform_(self.output_layer.bias)

    def forward(self, x=None):
        for layer in self.layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


def create_policy_net(game, config):
    input_dim = game.information_state_tensor_shape()[0]
    output_dim = game.num_distinct_actions()
    return PolicyNetwork(input_dim=input_dim, output_dim=output_dim, **config)


def get_nn_probs(nn_policy, state, player):
    information_state_tensor = torch.FloatTensor(state.information_state_tensor(player))
    logits = nn_policy(information_state_tensor)
    mask = torch.BoolTensor(state.legal_actions_mask(player))
    logits[~mask] = float("-inf")
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities


def nn_to_tabular_policy(game, nn_policy, player):
    tabular_policy = policy_module.TabularPolicy(game)
    info_sets_covered = set()
    for state in all_game_states(game, player):
        info_set_str = state.information_state_string(player)
        if info_set_str in info_sets_covered:
            continue
        info_sets_covered.add(info_set_str)

        action_probabilities = get_nn_probs(nn_policy, state, player).detach().numpy()

        for action, prob in enumerate(action_probabilities):
            tabular_policy.policy_for_key(info_set_str)[action] = prob
    return tabular_policy
