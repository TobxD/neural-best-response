import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class UnifiedPolicyGradientTrainer:
    def __init__(self, game, policy_network, train_params, hypernet_enabled=False):
        self.game = game
        self.policy_network = policy_network
        self.optimizer = optim.Adam(
            policy_network.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params["weight_decay"],
        )
        self.num_episodes = train_params["num_episodes"]
        self.epsilon = train_params.get("epsilon", 0)  # Default to 0 if not provided
        self.hypernet_enabled = hypernet_enabled

    def epsilon_random(self, probs, state, cur_player):
        uniform = torch.tensor(state.legal_actions_mask(cur_player))
        uniform = uniform / uniform.sum()
        return (1 - self.epsilon) * probs + self.epsilon * uniform

    def train_best_response(self, opponent_network, br_player_id):
        for episode in tqdm(range(self.num_episodes)):
            networks = (
                [self.policy_network, opponent_network]
                if br_player_id == 0
                else [opponent_network, self.policy_network]
            )

            state = self.game.new_initial_state()
            done = False
            reward = 0
            log_probs = []
            is_ratios = []

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
                        probs_function = (
                            get_hypernet_probs
                            if self.hypernet_enabled
                            else get_nn_probs
                        )
                        probs = probs_function(networks[cur_player], state, cur_player)

                        if cur_player == br_player_id:
                            orig_probs = probs.clone()
                            probs = self.epsilon_random(probs, state, cur_player)

                        m = torch.distributions.Categorical(probs)
                        action = m.sample()
                        if cur_player == br_player_id:
                            log_prob = torch.log(orig_probs[action])
                            log_probs.append(log_prob)
                            if self.hypernet_enabled:
                                is_ratios.append(orig_probs[action] / probs[action])
                        actions.append(action.item())

                if len(actions) == 1:
                    state.apply_action_with_legality_check(actions[0])
                else:
                    state.apply_actions_with_legality_checks(actions)

                if state.is_terminal():
                    done = True
                    reward = state.returns()[br_player_id]

            log_probs = torch.stack(log_probs)
            is_ratios = torch.stack(is_ratios).detach()

            loss = -log_probs * reward * is_ratios
            loss = loss.sum()

            # maybe sometimes nan or inf if the action we sampled with epsilon greedy has 0 support in the actual policy
            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
