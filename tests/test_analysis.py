import numpy as np

from analysis import run_detailed_evaluation
from fix_env import HIT, STAND, BlackjackEnvironment


class FixedAgent:
    """Always takes the same action"""

    def __init__(self, action):
        self.action = action

    def choose_action(self, state):
        return self.action


def test_logged_actions_use_the_correct_labels():
    np.random.seed(0)
    env = BlackjackEnvironment()
    stand_games, _ = run_detailed_evaluation(FixedAgent(STAND), env, num_games=20)
    hit_games, _ = run_detailed_evaluation(FixedAgent(HIT), env, num_games=20)
    assert {g["player_actions"] for g in stand_games} == {"STAND"}
    assert all(set(g["player_actions"].split(" -> ")) == {"HIT"} for g in hit_games)


def test_final_scores_and_busts_are_recorded():
    np.random.seed(0)
    env = BlackjackEnvironment()
    games, stats = run_detailed_evaluation(FixedAgent(HIT), env, num_games=50)
    # Always hitting ends in a bust unless the player reaches exactly 21
    assert all(g["player_final_score"] >= 21 for g in games)
    assert stats["player_busts"] > 0
    assert all(g["dealer_final_score"] > 0 for g in games)
