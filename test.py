import random
import pyspiel
import numpy as np
from pprint import pprint

# from pyspiel import exploitability
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
import copy

pprint(pyspiel.registered_names())
n = 3
util = np.random.rand(3, 3) * 2 - 1
print(util)
game = pyspiel.create_matrix_game(util, -util)
print(game)
print(game.num_distinct_actions())
state = game.new_initial_state()
print(state)
# rnd_strategy = np.ones(game.num_distinct_actions()) / game.num_distinct_actions()
# policy = pyspiel.TabularPolicy(game)
# rnd_policy = pyspiel.UniformRandomPolicy(game)
# conv = exploitability.nash_conv(game, rnd_policy)


print("....")


def action_probabilities(state):
    print(state)
    print(state.legal_actions(), state.num_distinct_actions())
    return {
        action: 1 / state.num_distinct_actions() for action in state.legal_actions()
    }


# p = policy.tabular_policy_from_callable(game, action_probabilities)
p = policy.TabularPolicy(game)
print(p.states_per_player)
print("state str", p.states[0].information_state_string(0))
print(p.policy_for_key(p.states_per_player[0][0]))

print(exploitability.nash_conv(game, p, False))
# print(p.action_probability_array)

br = best_response.BestResponsePolicy(game, 1, p)

combined_policy = policy.TabularPolicy(game)
combined_policy.action_probability_array[0] = p.action_probability_array[0]
combined_policy.action_probability_array[1] = 0
for s in combined_policy.states_per_player[1]:
    combined_policy.policy_for_key(s)[br.best_response_action(s)] = 1
# print("combined", combined_policy.action_probability_array)

# print(br)
print(expected_game_score.policy_value(game.new_initial_state(), p))
print(expected_game_score.policy_value(game.new_initial_state(), combined_policy))
