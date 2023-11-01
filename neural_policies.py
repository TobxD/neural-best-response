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

        layers = [input_dim] + layers + [output_dim]
        for num_in, num_out in zip(layers[:-1], layers[1:]):
            linear_layer = nn.Linear(num_in, num_out)
            init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
            # we can't use just 0 bias if we have 0 input for NF games
            init.uniform_(linear_layer.bias)
            self.layers.append(linear_layer)

    def forward(self, x=None):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

    def num_weight_values(self):
        return sum([layer.weight.numel() + layer.bias.numel() for layer in self.layers])

    def get_weights(self):
        weights = torch.zeros(self.num_weight_values())
        cnt = 0
        for layer in self.layers:
            num_weights = layer.weight.numel()
            weights[cnt : cnt + num_weights] = layer.weight.data.view(-1)
            cnt += num_weights
            num_bias = layer.bias.numel()
            weights[cnt : cnt + num_bias] = layer.bias.data.view(-1)
            cnt += num_bias
        return weights.detach()

    def set_weights(self, weights):
        """
        weights is a 1d tensor
        """
        assert weights.shape == (self.num_weight_values(),)
        cnt = 0
        for layer in self.layers:
            num_weights = layer.weight.numel()
            layer.weight.data = weights[cnt : cnt + num_weights].view(
                layer.weight.shape
            )
            cnt += num_weights
            num_bias = layer.bias.numel()
            layer.bias.data = weights[cnt : cnt + num_bias].view(layer.bias.shape)
            cnt += num_bias


class HyperNetworkActionOutput(nn.Module):
    def __init__(self, input_nn, input_dim, output_dim, layers):
        super(HyperNetworkActionOutput, self).__init__()
        input_dim += input_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, layers)

    def forward(self, model, x):
        model_weights = model.get_weights()
        x = torch.cat([model_weights.detach(), x.detach()], dim=-1)
        return self._model(x)


class HyperNetworkNNOutput(nn.Module):
    def __init__(self, input_nn, output_nn, layers):
        super(HyperNetworkNNOutput, self).__init__()
        input_dim = input_nn.num_weight_values()
        output_dim = output_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, layers)
        self._output_model_clone = output_nn.clone()

    def model_output(self, in_model):
        model_weights = in_model.get_weights()
        x = torch.cat([model_weights.detach(), x.detach()], dim=-1)
        return self._model(x)

    def forward(self, in_model, x):
        """
        forward pass of the hypernetwork and the output network
        """
        out_nn_weights = self.model_output(in_model)
        out_nn = self._output_model_clone.clone()
        out_nn.set_weights(out_nn_weights)
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
    net = HyperNetworkNNOutput(input_net, output_net**config)
    return net


def get_nn_probs(nn_policy, state, player):
    information_state_tensor = torch.FloatTensor(state.information_state_tensor(player))
    logits = nn_policy(information_state_tensor)
    mask = torch.BoolTensor(state.legal_actions_mask(player))
    logits = logits.masked_fill(~mask, float("-inf"))
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities


def get_hypernet_probs(hypernet, input_net, state, player):
    information_state_tensor = torch.FloatTensor(state.information_state_tensor(player))
    logits = hypernet(input_net, information_state_tensor)
    mask = torch.BoolTensor(state.legal_actions_mask(player))
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

        if type(nn_policy) == HyperNetworkActionOutput:
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
