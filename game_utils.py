from itertools import product
import numpy as np
import pyspiel
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import (
    expected_game_score,
    best_response,
    exploitability,
    cfr,
)
import nashpy


def all_game_states(game, player):
    player_states = []

    def _traverse(state):
        if state.is_terminal():
            return
        # player is -2 if simulataneous game (the only state in NF games)
        if state.current_player() == player or state.current_player() == -2:
            player_states.append(state.clone())
        for action in state.legal_actions():
            next_state = state.clone()
            next_state.apply_action_with_legality_check(action)
            _traverse(next_state)

    initial_state = game.new_initial_state()
    _traverse(initial_state)

    return player_states


def create_random_normal_game(num_actions):
    util = np.random.rand(num_actions, num_actions) * 2 - 1
    actions_a = [f"a{i+1}" for i in range(num_actions)]
    actions_b = [f"b{i+1}" for i in range(num_actions)]
    game = pyspiel.create_matrix_game(
        "random NF game", "random normal-form game", actions_a, actions_b, util, -util
    )
    return game


def matrix_nash_eqilibrium(game):
    utility_matrix_row = game.row_utilities()
    utility_matrix_col = game.col_utilities()

    # Use nashpy to find Nash equilibrium
    game_nash = nashpy.Game(utility_matrix_row, utility_matrix_col)
    strategy_p0, strategy_p1 = game_nash.linear_program()

    # Create TabularPolicy for both players based on the equilibrium
    tabular_policy = policy_module.TabularPolicy(game)
    state = tabular_policy.states_per_player[0][0]
    for action, prob in enumerate(strategy_p0):
        tabular_policy.policy_for_key(state)[action] = prob
    state = tabular_policy.states_per_player[1][0]
    for action, prob in enumerate(strategy_p1):
        tabular_policy.policy_for_key(state)[action] = prob
    return tabular_policy


def nash_equilibrium_policy_and_value(game, num_iters=10):
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        policy = matrix_nash_eqilibrium(game)
    else:
        cfr_solver = cfr.CFRSolver(game)
        for _ in range(num_iters):
            cfr_solver.evaluate_and_update_policy()

        conv = exploitability.exploitability(game, cfr_solver.average_policy())
        print("Exploitability:", conv)
        policy = cfr_solver.average_policy()
        print(policy)
    return policy, policy_value(game, policy)


def compute_best_response_tabular_policy(
    game, policy, player_id: int
) -> policy_module.TabularPolicy:
    """
    policy: policy to respond to
    player_id: player id of the best responder
    """
    br = best_response.BestResponsePolicy(game, player_id, policy)
    combined_policy = policy_module.TabularPolicy(game)
    for s in combined_policy.states_per_player[player_id]:
        combined_policy.policy_for_key(s)[:] = 0
        combined_policy.policy_for_key(s)[br.best_response_action(s)] = 1
    return combined_policy

def combine_tabular_policies(
    game, policy_a: policy_module.TabularPolicy, policy_b: policy_module.TabularPolicy
):
    combined_policy = policy_module.TabularPolicy(game)
    individual_policies = [policy_a, policy_b]
    for player_id in [0, 1]:
        for s in combined_policy.states_per_player[player_id]:
            combined_policy.policy_for_key(s)[:] = individual_policies[
                player_id
            ].policy_for_key(s)
    return combined_policy


def policy_value(game, policy):
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        return expected_game_score.policy_value(game.new_initial_state(), policy)
    else:
        return expected_game_score.policy_value(
            game.new_initial_state(), (policy, policy)
        )

def best_fixed_response_kuhn(game, opponent_policies, br_player_id):
    br_policies = np.array(list(product([0, 1], repeat=6)))
    # br_policies = np.array([[0, 0, 0, 0, 0, 1]])
    res = {}
    for br_policy in br_policies:
        probs = np.concatenate([br_policy[:, None], 1-br_policy[:, None]], axis=-1)
        tabular_response_policy = policy_module.TabularPolicy(game)
        if br_player_id == 0:
            tabular_response_policy.action_probability_array[:6] = probs
        else:
            tabular_response_policy.action_probability_array[6:] = probs
        values = []
        for opponent_policy in opponent_policies:
            tabular_policies = [tabular_response_policy, opponent_policy] if br_player_id == 0 else [opponent_policy, tabular_response_policy]
            combined_policy = combine_tabular_policies(game, *tabular_policies)
            values.append(policy_value(game, combined_policy)[br_player_id])
        res[tuple(br_policy.tolist())] = sum(values)/len(values)
    return res
