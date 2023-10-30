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
            next_state.apply_action(action)
            _traverse(next_state)

    initial_state = game.new_initial_state()
    _traverse(initial_state)

    return player_states
