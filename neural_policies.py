import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from open_spiel.python import policy as policy_module
import copy
from datetime import datetime

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
            if False and activation == "relu":
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

    def batched_model_inference(self, weights, x):
        cur_weight_ind = 0
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                num_weight_params = layer.weight.numel()
                layer_weights = weights[:, cur_weight_ind : cur_weight_ind + num_weight_params]
                cur_weight_ind += num_weight_params
                num_bias_params = layer.bias.numel()
                layer_bias = weights[:, cur_weight_ind : cur_weight_ind + num_bias_params]
                cur_weight_ind += num_bias_params

                # layer_bias: (batch_size, num_outputs)
                # layer_weights: (batch_size, num_outputs*num_inputs)
                layer_weights = layer_weights.view(
                    layer_weights.shape[0], layer.weight.shape[0], layer.weight.shape[1]
                )
                layer_bias = layer_bias.view(layer_bias.shape[0], layer.bias.shape[0])

                x = torch.bmm(layer_weights, x.unsqueeze(-1)).squeeze(-1)
                x = x + layer_bias
            else:
                x = layer(x)
        return x


class HyperNetworkActionOutput(nn.Module):
    def __init__(self, input_nn, input_dim, output_dim, **kwargs):
        super(HyperNetworkActionOutput, self).__init__()
        input_dim += input_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, **kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, model, x, model_weights=None):
        x = 2 * x - 1
        if model_weights is None:
            if not isinstance(model, list):
                model = [model]
            model_weights = [m.get_weights() for m in model]
        model_weights = torch.stack(model_weights, dim=0)
        got_weights_time = datetime.now()

        model_weights = model_weights.to(self.device)
        x = x.to(self.device)

        if len(model_weights.shape) > len(x.shape):
            model_weights = model_weights.squeeze(0)
        elif model_weights.shape[0] < x.shape[0]:
            model_weights = model_weights.repeat(x.shape[0], 1)
        x = torch.cat([model_weights.detach(), x.detach()], dim=-1)
        res = self._model(x)
        return res


class HyperNetworkNNOutput(nn.Module):
    def __init__(self, input_nn, output_nn, **kwargs):
        super(HyperNetworkNNOutput, self).__init__()
        input_dim = input_nn.num_weight_values()
        output_dim = output_nn.num_weight_values()
        self._model = PolicyNetwork(input_dim, output_dim, **kwargs)
        self._output_model_clone = copy.deepcopy(output_nn)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self._output_model_clone.to(self.device)

    def model_weight_output(self, in_model, model_weights=None):
        if model_weights is None:
            if not isinstance(in_model, list):
                in_model = [in_model]
            model_weights = [m.get_weights() for m in in_model]
        model_weights = torch.stack(model_weights, dim=0)
        model_weights = model_weights.to(self.device)

        out_nn_weights = self._model(model_weights.detach())
        return out_nn_weights

    def model_output(self, in_model, model_weights=None):
        start_time = datetime.now()
        out_nn_weights = self.model_weight_output(in_model, model_weights=model_weights)
        weight_time = datetime.now()
        # weight_norm = torch.linalg.norm(out_nn_weights, ord=1)
        # max_norm = out_nn_weights.numel() * 2
        # if weight_norm > max_norm:
        #     out_nn_weights = out_nn_weights / weight_norm * max_norm
        out_nn_weights = out_nn_weights.to("cpu")
        self._output_model_clone.to("cpu")
        move_time = datetime.now()
        res = []
        clone_time = datetime.now() - datetime.now()
        for i in range(out_nn_weights.shape[0]):
            start_clone = datetime.now()
            out_nn = copy.deepcopy(self._output_model_clone)
            end_clone = datetime.now()
            clone_time += end_clone - start_clone
            # out_nn.to(self.device)
            # out_nn.to("cpu")
            out_nn.set_weights(out_nn_weights[i])
            res.append(out_nn)
        end_time = datetime.now()
        # print(f"weight time: {weight_time - start_time},\nmove time: {move_time - weight_time},\nconvert time: {end_time - move_time}\nclone time: {clone_time}")
        return res

    def forward(self, in_model, x, model_weights=None):
        """
        forward pass of both the hypernetwork and the output network
        """
        if isinstance(in_model, list):
            return self.batched_forward(in_model, x, model_weights=model_weights)
        start_time = datetime.now()
        out_nn = self.model_output(in_model, model_weights=model_weights)
        got_model_time = datetime.now()
        action_logits = []
        # x = x.to(self.device)
        x = x.to("cpu")
        for i, nn in enumerate(out_nn):
            inp = x if len(x) > len(out_nn) else x[i]
            action_logits.append(nn(inp))
        end_time = datetime.now()
        # print(f"hypernet time: {got_model_time - start_time},\nnn time: {end_time - got_model_time}\n")
        if isinstance(in_model, list):
            action_logits = torch.stack(action_logits, dim=0)
        else:
            action_logits = action_logits[0]
        return action_logits.to(self.device)

    def batched_forward(self, in_model, x, model_weights=None):
        x = x.to(self.device)
        out_nn_weights = self.model_weight_output(in_model, model_weights=model_weights)
        res = self._output_model_clone.batched_model_inference(out_nn_weights, x)
        return res


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
    if isinstance(state, list):
        information_state_tensor = torch.FloatTensor([
            s.information_state_tensor(player) for s in state
        ])
    else:
        information_state_tensor = torch.FloatTensor(state.information_state_tensor(player))
    logits = nn_policy(information_state_tensor)
    if isinstance(state, list):
        mask = torch.BoolTensor([
            s.legal_actions_mask(player) for s in state
        ])
    else:
        mask = torch.BoolTensor(state.legal_actions_mask(player))
    logits = logits.masked_fill(~mask, float("-inf"))
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities

def get_hypernet_output(
    hypernet,
    input_net,
    state,
    player,
    information_state_tensor=None,
    model_weights=None,
):
    if information_state_tensor is None:
        if isinstance(state, list):
            information_state_tensor = [s.information_state_tensor(player) for s in state]
        else:
            information_state_tensor = state.information_state_tensor(player)
    information_state_tensor = torch.FloatTensor(information_state_tensor)
    res = hypernet(input_net, information_state_tensor, model_weights=model_weights)
    return res

def get_hypernet_probs(
    hypernet,
    input_net,
    state,
    player,
    information_state_tensor=None,
    legal_actions_mask=None,
    softmax_temp=1,
    model_weights=None,
):
    if legal_actions_mask is None:
        if isinstance(state, list):
            legal_actions_mask = [
                s.legal_actions_mask(player) for s in state
            ]
        else:
            legal_actions_mask = state.legal_actions_mask(player)
    logits = get_hypernet_output(
        hypernet,
        input_net,
        state,
        player,
        information_state_tensor,
        model_weights=model_weights,
    )
    mask = torch.tensor(legal_actions_mask, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    logits *= 1/softmax_temp
    action_probabilities = F.softmax(logits, dim=-1)
    return action_probabilities


def nn_to_tabular_policy(game, nn_policy, player, input_net=None):
    tabular_policy = policy_module.TabularPolicy(game)
    info_sets_covered = set()
    unique_states = []
    for state in all_game_states(game, player):
        info_set_str = state.information_state_string(player)
        if info_set_str in info_sets_covered:
            continue
        info_sets_covered.add(info_set_str)
        unique_states.append(state)

    if (
        type(nn_policy) == HyperNetworkActionOutput
        or type(nn_policy) == HyperNetworkNNOutput
    ):
        action_probabilities = (
            get_hypernet_probs(nn_policy, input_net, unique_states, player).cpu().detach().numpy()
        )
    else:
        action_probabilities = (
            get_nn_probs(nn_policy, unique_states, player).detach().numpy()
        )

    for si, state in enumerate(unique_states):
        info_set_str = state.information_state_string(player)
        for action, prob in enumerate(action_probabilities[si]):
            tabular_policy.policy_for_key(info_set_str)[action] = prob
    return tabular_policy
