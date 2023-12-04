from collections import defaultdict
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from pprint import pprint
from datetime import datetime
import sys
import os

import yaml

from neural_policies import create_hypernet_actionoutput, create_policy_net, get_hypernet_output, get_hypernet_probs, get_nn_probs
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
    def __init__(self, game, policy_network, train_params, input_net):
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

        q_net_config = yaml.safe_load(open(train_params['q_net_config'], "r"))
        self.q_network = create_hypernet_actionoutput(game, input_net, q_net_config)
        self.q_optim = optim.Adam(
            self.q_network.parameters(),
            lr=train_params["q_learning_rate"],
            weight_decay=train_params["q_weight_decay"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)

    def epsilon_random(self, probs, state, cur_player):
        uniform = torch.tensor(state.legal_actions_mask(cur_player))
        uniform = uniform / uniform.sum()
        res = (1 - self.epsilon) * probs + uniform * self.epsilon
        assert abs(res.sum() - 1.0) < 1e-6
        return res

    def run_k_episodes(self, game, networks, br_player_id, num_episodes, softmax_temp=1):
        states = [game.new_initial_state() for _ in range(num_episodes)]

        rewards = [None] * num_episodes
        br_actions = []
        log_probs = []
        ind = [[] for _ in range(num_episodes)]
        info_states = []
        legal_actions = []
        all_orig_probs = []
        # importance sampling ratios
        is_ratios = []

        to_play = [i for i in range(num_episodes)]

        while True:
            new_to_play = []
            current_to_play = []
            for episode in to_play:
                state = states[episode]
                if state.is_terminal():
                    if rewards[episode] is None:
                        rewards[episode] = state.returns()[br_player_id]
                    continue
                new_to_play.append(episode)
                if state.is_simultaneous_node() or state.current_player() == br_player_id:
                    current_to_play.append(episode)
            to_play = new_to_play
            if len(to_play) == 0:
                break

            if len(current_to_play) > 0:
                cur_input_states = [states[i] for i in current_to_play]
                nn_probs = get_hypernet_probs(
                    networks[br_player_id],
                    networks[1 - br_player_id],
                    cur_input_states,
                    br_player_id,
                    softmax_temp=softmax_temp,
                ).cpu()

            curind = 0
            for episode in to_play:
                state = states[episode]
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
                            probs = nn_probs[curind]
                            curind += 1
                            orig_probs = probs
                            probs = self.epsilon_random(probs, state, cur_player)
                        else:
                            probs = get_nn_probs(networks[cur_player], state, cur_player)
                        m = torch.distributions.Categorical(probs)
                        action = m.sample()
                        if cur_player == br_player_id:
                            ind[episode].append(len(log_probs))

                            log_prob = torch.log(orig_probs[action])
                            log_probs.append(log_prob)
                            is_ratios.append(orig_probs[action] / probs[action])
                            br_actions.append(action.item())
                            info_states.append(state.information_state_tensor(cur_player))
                            legal_actions.append(state.legal_actions_mask(cur_player))
                            all_orig_probs.append(orig_probs[action])
                        actions.append(action.item())

                if len(actions) == 1:
                    state.apply_action_with_legality_check(actions[0])
                else:
                    state.apply_actions_with_legality_checks(actions)


        log_probs = torch.stack(log_probs)
        is_ratios = torch.stack(is_ratios).detach()
        # is_ratios = 1
        action_rewards = [None] * len(log_probs)
        for episode in range(num_episodes):
            for i in ind[episode]:
                action_rewards[i] = rewards[episode]
        action_rewards = torch.tensor(action_rewards)

        return {
            "info_states": info_states,
            "legal_actions": legal_actions,
            "actions": br_actions,
            "log_probs": log_probs,
            "rewards": action_rewards,
            "is_ratios": is_ratios,
            "orig_probs": all_orig_probs,
            "ind": ind,
        }

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
                        ).cpu()
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
        table_br_value = main.policy_value(self.game, combined_policy)
        print("value of table br policy", table_br_value)

        nn_br = neural_policies.nn_to_tabular_policy(
            self.game, hypernet, hypernet_player, opponent_net
        )
        print(nn_br.action_probability_array)
        policies = (
            [opponent_tab, nn_br] if opponent_player == 0 else [nn_br, opponent_tab]
        )
        combined_policy = main.combine_tabular_policies(self.game, *policies)
        nn_br_value = main.policy_value(self.game, combined_policy)
        print(
            "value of hypernet br policy", nn_br_value
        )

        pure_strat = copy.deepcopy(nn_br)
        for i in range(len(pure_strat.action_probability_array)):
            if pure_strat.action_probability_array[i][0] > 0.5:
                pure_strat.action_probability_array[i][0] = 1
            else:
                pure_strat.action_probability_array[i][0] = 0
            pure_strat.action_probability_array[i][1] = 1 - pure_strat.action_probability_array[i][0]
        policies = (
            [opponent_tab, pure_strat] if opponent_player == 0 else [pure_strat, opponent_tab]
        )
        combined_policy = main.combine_tabular_policies(self.game, *policies)
        nn_pure_br_value = main.policy_value(self.game, combined_policy)
        print(
            "value of pure br policy", nn_pure_br_value
        )
        return table_br_value[hypernet_player], nn_br_value[hypernet_player], nn_pure_br_value[hypernet_player]

    def train_best_response(self, opponent_config, br_player_id):
        def lr_lambda(current_step):
            for step, lr_change in zip(self.train_params["lr_decay_steps"], self.train_params["lr_decay_factors"]):
                if current_step < step:
                    return 1/lr_change
            return 1/self.train_params["lr_decay_factors"][-1]
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        input_networks = read_all_nn(self.game, self.train_params["input_net_folder"])
        eval_networks = input_networks[4500:]
        input_networks = input_networks[:4500]
        baseline = defaultdict(lambda: (0, 0))
        loss_per_action = defaultdict(lambda : [[], []])
        cur_time = datetime.now().strftime("%Y%m%d_%H%M")
        if not os.path.exists(f'trajectory/{cur_time}'):
                os.makedirs(f'trajectory/{cur_time}')
        with open(f'trajectory/{cur_time}/train_config.yaml', 'w') as file:
            yaml.dump(self.train_params, file)
        
        for episode in tqdm(range(self.num_episodes)):
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

            num_episodes_batch = 100
            exp_per_state = defaultdict(lambda : [[], []])
            experiences = self.run_k_episodes(self.game, networks, br_player_id, num_episodes_batch)
            for e in range(len(experiences['rewards'])):
                dict_key = tuple(experiences["info_states"][e])
                base = baseline[dict_key]
                base = (base[0] + experiences['rewards'][e], base[1] + 1)
                baseline[dict_key] = base
                loss_per_action[dict_key][experiences['actions'][e]].append(experiences['rewards'][e])
                exp_per_state[dict_key][experiences['actions'][e]].append((experiences['log_probs'][e], experiences['rewards'][e]))

            loss = 0
            states = list(exp_per_state.keys())
            for s in states:
                avg_r0 = sum(x[1] for x in exp_per_state[s][0]) / max(1, len(exp_per_state[s][0]))
                avg_r1 = sum(x[1] for x in exp_per_state[s][1]) / max(1, len(exp_per_state[s][1]))
                # avg_r0 = sum(loss_per_action[s][0]) / max(1, len(loss_per_action[s][0]))
                # avg_r1 = sum(loss_per_action[s][1]) / max(1, len(loss_per_action[s][1]))
                # base = min(avg_r0, avg_r1)
                base = (avg_r0 + avg_r1) / 2
                for a in [0, 1]:
                    if len(exp_per_state[s][a]) == 0:
                        continue
                    log_probs, reward = zip(*exp_per_state[s][a])
                    log_probs = torch.stack(log_probs).squeeze()
                    reward = torch.tensor(reward)
                    # regret matching policy gradient (RMPG)
                    # loss += (-torch.exp(log_probs) * torch.clip(reward - base, -1e-9, torch.inf)).sum()
                    # standard policy gradient (REINFORCE with baseline)
                    # loss += (-log_probs * (reward - base)).sum()
                    # loss under uniform policy
                    loss += (-torch.exp(log_probs) * (reward - base)).sum()
            loss /= num_episodes_batch

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            filename = f'trajectory/{cur_time}/train_evaluation_log.txt'
            filename2 = f'trajectory/{cur_time}/eval_evaluation_log.txt'
    
            """eval training"""
            if episode % 1000 == 0:
                total_true, total_nn, total_pure = 0, 0, 0
                input_network_count = len(input_networks)

                with open(filename, 'a') as file:
                    original_stdout = sys.stdout
                    sys.stdout = file
                    
                    print('episode', episode)
                    for input_network in input_networks:
                        sys.stdout = None
                        t,n,p = self.eval_network(
                            input_network,
                            self.policy_network,
                            br_player_id
                        )
                        total_true += t
                        total_nn += n
                        total_pure += p
                        
                    sys.stdout = file
                    print('average true value',total_true/input_network_count)
                    print('average nn value',total_nn/input_network_count)
                    print('average pure value',total_pure/input_network_count)
                    sys.stdout = original_stdout

                """eval test"""
                if episode % 500 == 0:
                    total_true, total_nn, total_pure = 0, 0, 0
                    input_network_count = len(eval_networks)

                    with open(filename2, 'a') as file:
                        original_stdout = sys.stdout
                        sys.stdout = file
                        
                        print('episode', episode)
                        for input_network in eval_networks:
                            sys.stdout = None
                            t,n,p = self.eval_network(
                                input_network,
                                self.policy_network,
                                br_player_id
                            )
                            total_true += t
                            total_nn += n
                            total_pure += p
                            
                        sys.stdout = file
                        print('average true value',total_true/input_network_count)
                        print('average nn value',total_nn/input_network_count)
                        print('average pure value',total_pure/input_network_count)
                        sys.stdout = original_stdout

            continue
            
    def train_best_response_q_baseline(self, opponent_config, br_player_id):
        # buffer for q-network training, saving (opponent net index, state, legal actions, action, reward)
        q_replay_buffer = ReplayBuffer(self.train_params["q_replay_buffer_size"])
        # buffer for policy network training
        policy_replay_buffer = ReplayBuffer(self.train_params["replay_buffer_size"])

        input_networks = read_all_nn(self.game, self.train_params["input_net_folder"])
        input_networks = input_networks[:4500]
        input_net_weights = [x.get_weights() for x in input_networks]
        baseline = defaultdict(lambda: (0, 0))
        loss_per_action = defaultdict(lambda : [[], []])
        filename = f'trajectory/evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        for episode in tqdm(range(self.num_episodes)):
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

            num_episodes_batch = 20
            exp_per_state = defaultdict(lambda : [[], []])
            experiences = self.run_k_episodes(self.game, networks, br_player_id, num_episodes_batch)
            for e in range(len(experiences['rewards'])):
                dict_key = tuple(experiences["info_states"][e])
                base = baseline[dict_key]
                base = (base[0] + experiences['rewards'][e], base[1] + 1)
                baseline[dict_key] = base
                loss_per_action[dict_key][experiences['actions'][e]].append(experiences['rewards'][e])
                exp_per_state[dict_key][experiences['actions'][e]].append((experiences['log_probs'][e], experiences['rewards'][e]))

                q_replay_buffer.add((input_net_ind, experiences["info_states"][e], experiences["legal_actions"][e], experiences["actions"][e], experiences["rewards"][e]))
                policy_replay_buffer.add((input_net_ind, experiences["info_states"][e], experiences["legal_actions"][e], experiences["actions"][e], experiences["rewards"][e], experiences["log_probs"][e]))

            if episode % self.train_params["train_every"] == 0 and len(q_replay_buffer) >= self.train_params['batch_size']:
                for _ in range(self.train_params['num_train_steps']):
                    start_time = datetime.now()
                    loss = 0
                    batch = policy_replay_buffer.sample(self.train_params['batch_size'])
                    input_nets, states, action_masks, actions, rewards, log_probs = map(list, zip(*batch))
                    input_weights = [input_net_weights[i] for i in input_nets]
                    input_nets = [input_networks[i] for i in input_nets]
                    rewards = torch.tensor(rewards, device=self.device)

                    sampled_inputs_time = datetime.now()

                    # q_vals = get_hypernet_output(self.q_network, input_nets, None, br_player_id, information_state_tensor=states).detach()
                    # base = q_vals.mean(dim=-1)
                    # base = q_vals.min(dim=-1).values
                    # probs = get_hypernet_probs(self.policy_network, input_nets, None, br_player_id, information_state_tensor=states, legal_actions_mask=action_masks)
                    probs = get_hypernet_probs(self.policy_network, input_nets, None, br_player_id, information_state_tensor=states, legal_actions_mask=action_masks, model_weights=input_weights)

                    got_probs_time = datetime.now()

                    entropy = -torch.log(probs) * probs
                    entropy = entropy.mean(dim=-1)
                    probs = probs[[i for i in range(len(actions))], actions]
                    # loss = -torch.log(probs) * torch.clip(rewards - base, -1e-9, torch.inf)
                    # loss = -torch.log(probs) * (rewards - base)
                    loss = -probs * rewards - self.train_params["entropy_penalty"] * entropy
                    loss = loss.mean()

                    loss_time = datetime.now()

                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_backward_time = datetime.now()
                    self.optimizer.step()
                    optimizer_step_time = datetime.now()

                    # print("sampled inputs time", sampled_inputs_time - start_time)
                    # print("got probs time", got_probs_time - sampled_inputs_time)
                    # print("loss time", loss_time - got_probs_time)
                    # print("loss backward time", loss_backward_time - loss_time)
                    # print("optimizer step time", optimizer_step_time - loss_backward_time)
                    # print()

            if episode % self.train_params["q_train_every"] == 0 and len(q_replay_buffer) >= self.train_params['q_batch_size']:
                batch = q_replay_buffer.sample(self.train_params['q_batch_size'])
                in_nets, states, _, actions, rewards = map(list, zip(*batch))
                in_nets = [input_networks[i] for i in in_nets]
                output_values = get_hypernet_output(self.q_network, in_nets, None, br_player_id, information_state_tensor=states)
                q_loss = 0
                for i in range(len(batch)):
                    q_loss += (output_values[i][actions[i]] - rewards[i]) ** 2
                q_loss /= len(batch)
                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

            """store the eval file with cur time"""
            if not os.path.exists('trajectory'):
                os.makedirs('trajectory')
            if episode % 275 == 0:
                total_true, total_nn, total_pure = 0, 0, 0
                input_network_count = len(input_networks)

                with open(filename, 'a') as file:
                    original_stdout = sys.stdout
                    sys.stdout = file
                    print('episode', episode)

                    sys.stdout = None
                    for input_network in input_networks:
                        t,n,p = self.eval_network(
                            input_network,
                            self.policy_network,
                            br_player_id
                        )
                        total_true += t
                        total_nn += n
                        total_pure += p
                        
                    sys.stdout = file
                    print('average true value',total_true/input_network_count)
                    print('average nn value',total_nn/input_network_count)
                    print('average pure value',total_pure/input_network_count)

                sys.stdout = original_stdout


            if not os.path.exists('trajectory'):
                os.makedirs('trajectory')
            if episode % 275 == 0:
                total_true, total_nn, total_pure = 0, 0, 0
                input_network_count = len(input_networks)

                with open(filename, 'a') as file:
                    original_stdout = sys.stdout
                    sys.stdout = file
                    print('episode', episode)

                    sys.stdout = None
                    for input_network in input_networks:
                        t,n,p = self.eval_network(
                            input_network,
                            self.policy_network,
                            br_player_id
                        )
                        total_true += t
                        total_nn += n
                        total_pure += p
                        
                    sys.stdout = file
                    print('average true value',total_true/input_network_count)
                    print('average nn value',total_nn/input_network_count)
                    print('average pure value',total_pure/input_network_count)

                sys.stdout = original_stdout
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
