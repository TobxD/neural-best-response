import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from open_spiel.python import policy as policy_module
import copy

from game_utils import all_game_states


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers, activation):
        super(PolicyNetwork, self).__init__()

        assert activation in ["relu", "tanh"]
        activation_module = nn.ReLU if activation == "relu" else nn.Tanh

        layers = [input_dim] + layers + [output_dim]
        layer_combinations = list(zip(layers[:-1], layers[1:]))
        network_layers = []
        for i, (num_in, num_out) in enumerate(layer_combinations):
            linear_layer = nn.Linear(num_in, num_out)
            init.kaiming_normal_(linear_layer.weight, nonlinearity=activation)
            if activation == "relu":
                linear_layer.bias.data.fill_(0.01)
            else:
                # we can't use just 0 bias if we have 0 input for NF games
                init.uniform_(linear_layer.bias)
            network_layers.append(linear_layer)
            if i < len(layer_combinations) - 1:
                network_layers.append(activation_module())
        self.network = nn.Sequential(*network_layers)

    def forward(self, x=None):
        return self.network(x)

    def num_weight_values(self):
        return sum([param.numel() for param in self.parameters()])

    def get_weights(self):
        weights = torch.zeros(self.num_weight_values())
        cnt = 0
        for param in self.parameters():
            num_weights = param.numel()
            weights[cnt : cnt + num_weights] = param.data.view(-1)
            cnt += num_weights
        return weights.detach()

    def set_weights(self, weights):
        """
        weights is a 1d tensor
        """
        assert weights.shape == (self.num_weight_values(),)
        cnt = 0
        for param in self.parameters():
            num_weights = param.numel()
            param.requires_grad = False
            param.copy_(weights[cnt : cnt + num_weights].view(param.shape))
            cnt += num_weights


class HyperNetworkActionOutput(nn.Module):
    def __init__(self, input_nn, input_dim, output_dim, **kwargs):
        super(HyperNetworkActionOutput, self).__init__()
        input_dim += input_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, **kwargs)

    def forward(self, model, x):
        model_weights = model.get_weights()
        x = torch.cat([model_weights.detach(), x.detach()], dim=-1)
        return self._model(x)


class HyperNetworkNNOutput(nn.Module):
    def __init__(self, input_nn, output_nn, **kwargs):
        super(HyperNetworkNNOutput, self).__init__()
        input_dim = input_nn.num_weight_values()
        output_dim = output_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, **kwargs)
        self._output_model_clone = copy.deepcopy(output_nn)

    def model_output(self, in_model):
        model_weights = in_model.get_weights()
        out_nn_weights = self._model(model_weights.detach())
        out_nn = copy.deepcopy(self._output_model_clone)
        out_nn.set_weights(out_nn_weights)
        return out_nn

    def forward(self, in_model, x):
        """
        forward pass of both the hypernetwork and the output network
        """
        out_nn = self.model_output(in_model)
        action_logits = out_nn(x)
        return action_logits


def create_policy_net(game, config):
    input_dim = game.information_state_tensor_shape()[0]
    output_dim = game.num_distinct_actions()
    net = PolicyNetwork(input_dim=input_dim, output_dim=output_dim, **config)
    return net


def create_hypernet_actionoutput(game, input_net, config):
    input_dim = game.information_state_tensor_shape()[0]
    output_dim = game.num_distinct_actions()
    net = HyperNetworkActionOutput(
        input_net, input_dim=input_dim, output_dim=output_dim, **config
    )
    return net


def create_hypernet_nn_output(game, input_net, config, output_net=None):
    if output_net is None:
        output_net = input_net
    net = HyperNetworkNNOutput(input_net, output_net, **config)
    return net


def get_nn_probs(nn_policy, state, player):
    information_state_tensor = torch.FloatTensor(state.information_state_tensor(player))
    logits = nn_policy(information_state_tensor)
    mask = torch.BoolTensor(state.legal_actions_mask(player))
    logits = logits.masked_fill(~mask, float("-inf"))
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities


def get_hypernet_probs(
    hypernet,
    input_net,
    state,
    player,
    information_state_tensor=None,
    legal_actions_mask=None,
):
    if information_state_tensor is None:
        information_state_tensor = state.information_state_tensor(player)
    information_state_tensor = torch.FloatTensor(information_state_tensor)
    if legal_actions_mask is None:
        legal_actions_mask = state.legal_actions_mask(player)
    logits = hypernet(input_net, information_state_tensor)
    mask = torch.BoolTensor(legal_actions_mask)
    logits = logits.masked_fill(~mask, float("-inf"))
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities


def nn_to_tabular_policy(game, nn_policy, player, input_net=None):
    tabular_policy = policy_module.TabularPolicy(game)
    info_sets_covered = set()
    for state in all_game_states(game, player):
        info_set_str = state.information_state_string(player)
        if info_set_str in info_sets_covered:
            continue
        info_sets_covered.add(info_set_str)

        if (
            type(nn_policy) == HyperNetworkActionOutput
            or type(nn_policy) == HyperNetworkNNOutput
        ):
            action_probabilities = (
                get_hypernet_probs(nn_policy, input_net, state, player).detach().numpy()
            )
        else:
            action_probabilities = (
                get_nn_probs(nn_policy, state, player).detach().numpy()
            )

        for action, prob in enumerate(action_probabilities):
            tabular_policy.policy_for_key(info_set_str)[action] = prob
    return tabular_policy
