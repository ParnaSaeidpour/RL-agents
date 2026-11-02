import numpy as np

from fix_env import HIT, STAND, BlackjackEnvironment


def test_action_ids_match_rlcard():
    env = BlackjackEnvironment()
    assert env.env.actions[HIT] == "hit"
    assert env.env.actions[STAND] == "stand"


def test_state_info_has_the_fields_the_agents_use():
    np.random.seed(0)
    env = BlackjackEnvironment()
    state, player = env.reset()
    info = env.get_state_info(state)
    assert player == 0
    assert 4 <= info["player_value"] <= 21
    assert set(info["legal_actions"]) == {HIT, STAND}


def test_dealer_hole_card_is_hidden_from_the_agent():
    env = BlackjackEnvironment()
    state, _ = env.reset()
    info = env.get_state_info(state)
    # The dealer starts with two cards; the agent only sees one of them
    assert len(info["dealer_hand"]) == 1
    assert len(env.env.game.dealer.hand) == 2


def test_standing_ends_the_game_with_a_payoff():
    env = BlackjackEnvironment()
    env.reset()
    env.step(STAND)
    assert env.is_over()
    assert env.get_payoffs()[0] in (-1, 0, 1)
