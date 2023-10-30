import numpy as np
import pyspiel
from open_spiel.python import policy as policy_module
from open_spiel.python.algorithms import (
    expected_game_score,
    best_response,
    exploitability,
    cfr,
)
import yaml
import neural_policies
import torch
import nashpy


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


def nash_eq_policy_and_value(game, num_iters=10):
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


def main(nn_config):
    game = create_random_normal_game(3)
    # game = pyspiel.load_game("kuhn_poker")
    print("nash", nash_eq_policy_and_value(game))
    print(game)
    np.random.seed(2)
    nn_player = 0
    nn_policy = neural_policies.create_policy_net(game, nn_config)
    nn_tab_policy = neural_policies.nn_to_tabular_policy(game, nn_policy, nn_player)
    print("nn tab policy", nn_tab_policy.action_probability_array)
    br = compute_best_response_tabular_policy(game, nn_tab_policy, 1 - nn_player)
    print("br policy", br.action_probability_array)
    policies = [nn_tab_policy, br] if nn_player == 0 else [br, nn_tab_policy]
    total_policy = combine_tabular_policies(game, *policies)
    print("total policy", total_policy.action_probability_array)
    print("nn policy value", policy_value(game, total_policy))


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    config = yaml.safe_load(open("code/config/mlp.yaml"))
    main(config)
