import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from open_spiel.python import policy as policy_module

from game_utils import all_game_states


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers, use_learned_input=False):
        super(PolicyNetwork, self).__init__()

        self.use_learned_input = use_learned_input
        self.layers = nn.ModuleList()

        # If learned parameters are used as input, initialize them
        if self.use_learned_input:
            self.learned_input = nn.Parameter(torch.randn(input_dim))

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
        print(self.use_learned_input, self.layers)
        if self.use_learned_input:
            x = self.learned_input

        for layer in self.layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


def create_policy_net(config):
    return PolicyNetwork(**config)


def get_nn_probs(nn_policy, state, player):
    information_state_tensor = torch.FloatTensor(state.observation_tensor(player))
    logits = nn_policy(information_state_tensor)
    possible_actions = state.legal_actions(player)

    mask = torch.zeros_like(logits)
    mask[possible_actions] = 1
    logits *= mask

    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities.detach().numpy()


def nn_to_tabular_policy(game, nn_policy, player):
    tabular_policy = policy_module.TabularPolicy(game)
    info_sets_covered = set()
    for state in all_game_states(game, player):
        info_set_str = state.information_state_string(player)
        if info_set_str in info_sets_covered:
            continue
        info_sets_covered.add(info_set_str)

        action_probabilities = get_nn_probs(nn_policy, state, player)
        print(action_probabilities)

        for action, prob in enumerate(action_probabilities):
            tabular_policy.policy_for_key(info_set_str)[action] = prob
    return tabular_policy


# def nn_to_tabular_policy(game, nn_policy, player):
#     tabular_policy = policy_module.TabularPolicy(game)

#     init_state = game.new_initial_state()

#     for state in tabular_policy.states_per_player[player]:
#         nn_probs = nn_policy().detach().numpy()
#         print("nn probs", nn_probs)
#         # assert len(nn_probs) == game.num_distinct_actions()
#         for action, prob in enumerate(nn_probs):
#             tabular_policy.policy_for_key(state)[action] = prob

#     return tabular_policy
