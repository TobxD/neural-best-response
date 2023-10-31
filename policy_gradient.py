import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from neural_policies import create_policy_net, get_hypernet_probs, get_nn_probs


class PolicyGradientTrainer:
    def __init__(self, game, policy_network, train_params):
        self.game = game
        self.policy_network = policy_network
        self.optimizer = optim.Adam(
            policy_network.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params["weight_decay"],
        )
        self.num_episodes = train_params["num_episodes"]

    def train_best_response(self, opponent_network, br_player_id):
        networks = (
            [self.policy_network, opponent_network]
            if br_player_id == 0
            else [
                opponent_network,
                self.policy_network,
            ]
        )
        for episode in range(self.num_episodes):
            state = self.game.new_initial_state()
            done = False
            reward = 0
            log_probs = []

            while not done:
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    actions = [np.random.choice(outcomes, p=probs)]
                else:
                    if state.is_simultaneous_node():
                        players = [0, 1]
                    else:
                        players = [state.current_player()]
                    actions = []
                    for cur_player in players:
                        probs = get_nn_probs(networks[cur_player], state, cur_player)
                        m = torch.distributions.Categorical(probs)
                        action = m.sample()
                        if cur_player == br_player_id:
                            log_prob = m.log_prob(action)
                            log_probs.append(log_prob)
                        actions.append(action.item())

                if len(actions) == 1:
                    state.apply_action_with_legality_check(actions[0])
                else:
                    state.apply_actions_with_legality_checks(actions)

                if state.is_terminal():
                    done = True
                    reward = state.returns()[br_player_id]

            log_probs = torch.stack(log_probs)

            loss = -log_probs * reward
            loss = loss.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class PolicyGradientHypernetTrainer:
    def __init__(self, game, policy_network, train_params):
        self.game = game
        self.policy_network = policy_network
        self.optimizer = optim.Adam(
            policy_network.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params["weight_decay"],
        )
        self.num_episodes = train_params["num_episodes"]

    def train_best_response(self, opponent_config, br_player_id):
        for episode in tqdm(range(self.num_episodes)):
            opponent_network = create_policy_net(self.game, opponent_config)
            networks = (
                [self.policy_network, opponent_network]
                if br_player_id == 0
                else [
                    opponent_network,
                    self.policy_network,
                ]
            )

            state = self.game.new_initial_state()
            done = False
            reward = 0
            log_probs = []

            while not done:
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    actions = [np.random.choice(outcomes, p=probs)]
                else:
                    if state.is_simultaneous_node():
                        players = [0, 1]
                    else:
                        players = [state.current_player()]
                    actions = []
                    for cur_player in players:
                        if cur_player == br_player_id:
                            probs = get_hypernet_probs(
                                networks[cur_player],
                                networks[1 - cur_player],
                                state,
                                cur_player,
                            )
                        else:
                            probs = get_nn_probs(
                                networks[cur_player], state, cur_player
                            )
                        m = torch.distributions.Categorical(probs)
                        action = m.sample()
                        if cur_player == br_player_id:
                            log_prob = m.log_prob(action)
                            log_probs.append(log_prob)
                        actions.append(action.item())

                if len(actions) == 1:
                    state.apply_action_with_legality_check(actions[0])
                else:
                    state.apply_actions_with_legality_checks(actions)

                if state.is_terminal():
                    done = True
                    reward = state.returns()[br_player_id]

            log_probs = torch.stack(log_probs)

            loss = -log_probs * reward
            loss = loss.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
