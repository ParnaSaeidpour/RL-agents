import numpy as np
import pytest

from fix_env import HIT, STAND
from mc import MonteCarloBlackjackAgent, calculate_hand_value_and_ace, decode_rlcard_state


@pytest.mark.parametrize(
    "hand, expected",
    [
        (["S7", "H9"], (16, False)),
        (["SA", "H6"], (17, True)),        # soft 17: ace counts as 11
        (["SA", "H6", "D9"], (16, False)),  # ace must count as 1
        (["SA", "HA"], (12, True)),
        (["SK", "HQ", "DA"], (21, False)),
        (["ST", "HJ"], (20, False)),
    ],
)
def test_hand_value_and_usable_ace(hand, expected):
    assert calculate_hand_value_and_ace(hand) == expected


def test_decode_uses_the_visible_dealer_card():
    raw = {"player0 hand": ["SA", "H6"], "dealer hand": ["DK"]}
    assert decode_rlcard_state(raw) == (17, 10, True)


def test_first_visit_update_uses_discounted_returns():
    agent = MonteCarloBlackjackAgent(gamma=0.5, alpha=1.0, first_visit=True)
    states = [(12, 10, False), (16, 10, False)]
    agent.update_q_values(states, [HIT, STAND], rewards=[0, 1])
    # Returns: G1 = 1, G0 = 0 + 0.5 * 1
    assert agent.Q[(16, 10, False)][STAND] == pytest.approx(1.0)
    assert agent.Q[(12, 10, False)][HIT] == pytest.approx(0.5)


def test_first_visit_counts_a_repeated_pair_once():
    agent = MonteCarloBlackjackAgent(gamma=1.0, alpha=1.0, first_visit=True)
    s = (12, 10, False)
    agent.update_q_values([s, s], [HIT, HIT], rewards=[0, -1])
    assert agent.N[s][HIT] == 1


def test_policy_follows_the_highest_q_value():
    agent = MonteCarloBlackjackAgent()
    agent.Q = {(19, 6, False): {HIT: -0.5, STAND: 0.4}, (9, 6, False): {HIT: 0.3, STAND: -0.1}}
    agent.extract_policy()
    assert agent.get_action(19, 6) == STAND
    assert agent.get_action(9, 6) == HIT


def test_unknown_states_fall_back_to_hit_below_17():
    agent = MonteCarloBlackjackAgent()
    assert agent.get_action(12, "K") == HIT
    assert agent.get_action(17, "A") == STAND


def test_greedy_policy_without_exploration_is_deterministic():
    np.random.seed(0)
    agent = MonteCarloBlackjackAgent(epsilon=0.0)
    agent.Q[(15, 7, False)] = {HIT: 0.2, STAND: -0.3}
    agent.N[(15, 7, False)] = {HIT: 0, STAND: 0}
    assert {agent.epsilon_greedy_policy((15, 7, False)) for _ in range(20)} == {HIT}
