from collections import defaultdict
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from pprint import pprint

from neural_policies import create_policy_net, get_hypernet_probs, get_nn_probs
import neural_policies
from input_net_data import read_all_nn
from open_spiel.python.utils.replay_buffer import ReplayBuffer
import main


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
        self.epsilon = train_params["epsilon"]
        self.train_params = train_params

    def epsilon_random(self, probs, state, cur_player):
        uniform = torch.tensor(state.legal_actions_mask(cur_player))
        uniform = uniform / uniform.sum()
        res = (1 - self.epsilon) * probs + uniform * self.epsilon
        assert abs(res.sum() - 1.0) < 1e-6
        return res

    def run_episode(self, game, networks, br_player_id):
        state = game.new_initial_state()
        done = False
        reward = 0
        log_probs = []
        # importance sampling ratios
        is_ratios = []
        experience = []

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
                        orig_probs = probs
                        probs = self.epsilon_random(probs, state, cur_player)
                    else:
                        probs = get_nn_probs(networks[cur_player], state, cur_player)
                    m = torch.distributions.Categorical(probs)
                    action = m.sample()
                    if cur_player == br_player_id:
                        experience.append(
                            (
                                state.information_state_tensor(cur_player),
                                state.legal_actions_mask(cur_player),
                                action,
                                orig_probs[action],
                            )
                        )
                        log_prob = torch.log(orig_probs[action])
                        log_probs.append(log_prob)
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
        # is_ratios = 1

        return experience, log_probs, reward, is_ratios

    def eval_network(self, opponent_net, hypernet, hypernet_player):
        networks = (
            [hypernet, opponent_net]
            if hypernet_player == 0
            else [
                opponent_net,
                hypernet,
            ]
        )
        opponent_player = 1 - hypernet_player
        opponent_tab = neural_policies.nn_to_tabular_policy(
            self.game, opponent_net, opponent_player
        )
        print(opponent_tab.action_probability_array)
        tab_br = main.compute_best_response_tabular_policy(
            self.game, opponent_tab, hypernet_player
        )
        print(tab_br.action_probability_array)
        policies = (
            [opponent_tab, tab_br] if opponent_player == 0 else [tab_br, opponent_tab]
        )
        combined_policy = main.combine_tabular_policies(self.game, *policies)
        print("value of table br policy", main.policy_value(self.game, combined_policy))

        nn_br = neural_policies.nn_to_tabular_policy(
            self.game, hypernet, hypernet_player, opponent_net
        )
        print(nn_br.action_probability_array)
        policies = (
            [opponent_tab, nn_br] if opponent_player == 0 else [nn_br, opponent_tab]
        )
        combined_policy = main.combine_tabular_policies(self.game, *policies)
        print(
            "value of hypernet br policy", main.policy_value(self.game, combined_policy)
        )

    def train_best_response(self, opponent_config, br_player_id):
        input_networks = read_all_nn(self.game, self.train_params["input_net_folder"])
        input_networks = input_networks[:5]
        baseline = defaultdict(lambda: (0, 0))
        loss_per_action = defaultdict(lambda : [[], []])
        for episode in tqdm(range(self.num_episodes)):
            # opponent_network = create_policy_net(self.game, opponent_config)
            input_net_ind = np.random.randint(len(input_networks))
            opponent_network = input_networks[input_net_ind]
            networks = (
                [self.policy_network, opponent_network]
                if br_player_id == 0
                else [
                    opponent_network,
                    self.policy_network,
                ]
            )

            cnt_per = 100
            loss = 0
            exp_per_state = defaultdict(lambda : [[], []])
            for _ in range(cnt_per):
                experience, log_probs, reward, is_ratios = self.run_episode(
                    self.game, networks, br_player_id
                )

                dict_key = tuple(experience[0][0])
                base = baseline[dict_key]
                base = (base[0] + reward, base[1] + 1)
                baseline[dict_key] = base
                base_val = base[0] / base[1]
                loss_per_action[dict_key][experience[0][2].item()].append(reward)
                exp_per_state[dict_key][experience[0][2].item()].append((log_probs, reward))

                # loss = -log_probs * reward * is_ratios
                new_loss = -log_probs * (reward - base_val)
                new_loss = new_loss.sum()

                # maybe sometimes nan or inf if the action we sampled with epsilon greedy has 0 support in the actual policy
                if not torch.isfinite(new_loss):
                    continue
                loss += new_loss
            loss /= cnt_per
            # out_weights = self.policy_network.model_weight_output(opponent_network)
            # weight_norm = torch.linalg.norm(out_weights, ord=1) * 1e-1
            # loss += weight_norm
            # for s, p in loss_per_action.items():
            #     print(s, p[0], p[1])

            loss *= 0
            for s in exp_per_state:
                avg_r0 = sum(x[1] for x in exp_per_state[s][0]) / max(1, len(exp_per_state[s][0]))
                avg_r1 = sum(x[1] for x in exp_per_state[s][1]) / max(1, len(exp_per_state[s][1]))
                # avg_r0 = sum(loss_per_action[s][0]) / max(1, len(loss_per_action[s][0]))
                # avg_r1 = sum(loss_per_action[s][1]) / max(1, len(loss_per_action[s][1]))
                base = min(avg_r0, avg_r1)
                for a in [0, 1]:
                    if len(exp_per_state[s][a]) == 0:
                        continue
                    log_probs, reward = zip(*exp_per_state[s][a])
                    log_probs = torch.stack(log_probs).squeeze()
                    reward = torch.tensor(reward)
                    loss += (-log_probs * (reward - base)).sum()
            loss /= cnt_per

            # loss *= 0
            # cur_res = []
            # if episode > 1000:
            #     for s, p in loss_per_action.items():
            #         # if s[3] < 0.5:
            #         #     continue
            #         r0 = sum(p[0]) / max(1, len(p[0]))
            #         r1 = sum(p[1]) / max(1, len(p[1]))
            #         probs = get_hypernet_probs(
            #             self.policy_network,
            #             opponent_network,
            #             None,
            #             br_player_id,
            #             information_state_tensor=torch.FloatTensor(s),
            #             legal_actions_mask=torch.BoolTensor(2 * [True]),
            #         )
            #         # base = (r0+r1)/2
            #         base = min(r0, r1)
            #         l1 = -torch.log(probs[0]) * (r0-base)
            #         l2 = -torch.log(probs[1]) * (r1-base)
            #         cur_res.append((s, l1.item(), l2.item(), r0-base, r1-base))
            #         loss += l1 + l2

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=3.0)
            self.optimizer.step()
            continue

    def train_simultaneous_br(self, opponent_config):
        replay_buffer = [
            ReplayBuffer(self.train_params["replay_buffer_size"]) for _ in range(2)
        ]

        current_nets = [
            create_policy_net(self.game, opponent_config),
            create_policy_net(self.game, opponent_config),
        ]
        policy_nets = [
            copy.deepcopy(self.policy_network),
            copy.deepcopy(self.policy_network),
        ]
        optimizers = [
            optim.Adam(
                policy_nets[i].parameters(),
                lr=self.train_params["learning_rate"],
                weight_decay=self.train_params["weight_decay"],
            )
            for i in [0, 1]
        ]
        # optimizers = [
        #     optim.SGD(
        #         policy_nets[i].parameters(),
        #         lr=self.train_params["learning_rate"],
        #         momentum=0,
        #         weight_decay=self.train_params["weight_decay"],
        #     )
        #     for i in [0, 1]
        # ]
        for episode in tqdm(range(self.num_episodes)):
            train_player = episode % 2
            opponent_player = 1 - train_player

            if episode % 250 < 4:
                self.eval_network(
                    current_nets[opponent_player],
                    policy_nets[train_player],
                    train_player,
                )

            networks = [
                policy_nets[i] if i == train_player else current_nets[i] for i in [0, 1]
            ]

            experience, log_probs, reward, is_ratios = self.run_episode(
                self.game, networks, train_player
            )

            for state, legal_actions, action, orig_prob in experience:
                replay_buffer[train_player].add(
                    (
                        state,
                        legal_actions,
                        action,
                        orig_prob,
                        reward,
                        copy.deepcopy(current_nets[opponent_player]),
                    )
                )

            if episode % self.train_params["train_every"] < 2:
                batch = replay_buffer[train_player].sample(
                    min(
                        len(replay_buffer[train_player]),
                        self.train_params["batch_size"],
                    )
                )
                loss = []
                flag = False
                if flag:
                    res = []
                    for (
                        state,
                        legal_actions,
                        action,
                        orig_prob,
                        reward,
                        opponent_net,
                    ) in batch:
                        new_probs = get_hypernet_probs(
                            policy_nets[train_player],
                            opponent_net,
                            None,
                            train_player,
                            information_state_tensor=state,
                            legal_actions_mask=legal_actions,
                        )
                        res.append(new_probs[action])
                for (
                    state,
                    legal_actions,
                    action,
                    orig_prob,
                    reward,
                    opponent_net,
                ) in batch:
                    new_probs = get_hypernet_probs(
                        policy_nets[train_player],
                        opponent_net,
                        None,
                        train_player,
                        information_state_tensor=state,
                        legal_actions_mask=legal_actions,
                    )
                    ratio = new_probs[action] / orig_prob.detach()
                    new_loss = -torch.min(
                        ratio * reward, torch.clip(ratio, 0.95, 1.05) * reward
                    )
                    model_weights_out = policy_nets[train_player].model_weight_output(
                        opponent_net
                    )
                    # weight_decay_loss = (model_weights_out * model_weights_out).sum()
                    weight_decay_loss = torch.linalg.norm(model_weights_out, ord=1)
                    # new_loss += weight_decay_loss * 1e-4
                    if not torch.isfinite(new_loss) or orig_prob < 1e-8:
                        continue
                    loss.append(new_loss)
                loss = torch.stack(loss).mean()
                if not torch.isfinite(loss):
                    continue
                optimizers[train_player].zero_grad()
                loss.backward()
                optimizers[train_player].step()
                if flag:
                    res2 = []
                    res3 = []
                    for (
                        state,
                        legal_actions,
                        action,
                        orig_prob,
                        reward,
                        opponent_net,
                    ) in batch:
                        new_probs = get_hypernet_probs(
                            policy_nets[train_player],
                            opponent_net,
                            None,
                            train_player,
                            information_state_tensor=state,
                            legal_actions_mask=legal_actions,
                        )
                        res2.append(new_probs[action])
                        res3.append(reward)
                    all_res = [
                        (x[0].item(), x[1].item(), x[2]) for x in zip(res, res2, res3)
                    ]
                    pprint(all_res)
                    cur = 0
                    for a, b, r in all_res:
                        cur += (b - a) * r
                    print("benefit:", cur)

            # loss = -log_probs * reward * is_ratios
            # loss = loss.sum()
            # # maybe sometimes nan or inf if the action we sampled with epsilon greedy has 0 support in the actual policy
            # if not torch.isfinite(loss):
            #     continue

            # optimizers[train_player].zero_grad()
            # loss.backward()
            # optimizers[train_player].step()

            if episode % 1 == 0 and episode > 200:
                new_net = policy_nets[train_player].model_output(
                    current_nets[opponent_player]
                )
                # TODO detach?
                current_nets[train_player] = new_net
                continue

        return policy_nets

# policy_nets[0]._model.network[0].weight.data[3:, 3:] = 50*torch.tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
# policy_nets[0]._model.network[0].bias.data = torch.zeros(6)
