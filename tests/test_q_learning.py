import numpy as np
import pytest

from fix_env import HIT, STAND, BlackjackEnvironment
from q_learning import QLearningAgent, evaluate_agent, train_q_learning_with_env

S1 = {"player_value": 12, "dealer_showing": 10}
S2 = {"player_value": 18, "dealer_showing": 10}


def test_terminal_update_moves_q_towards_the_reward():
    agent = QLearningAgent(learning_rate=0.5)
    agent.update(S1, STAND, reward=1, next_state=S2, done=True)
    # Q = 0 + 0.5 * (1 - 0)
    assert agent.q_table[(12, 10)][STAND] == pytest.approx(0.5)
    assert agent.q_table[(12, 10)][HIT] == 0


def test_non_terminal_update_uses_the_best_next_action():
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
    agent.q_table[(18, 10)] = np.array([0.2, 0.6])
    agent.update(S1, HIT, reward=0, next_state=S2, done=False)
    # Q = 0 + 0.5 * (0 + 0.9 * max(0.2, 0.6) - 0)
    assert agent.q_table[(12, 10)][HIT] == pytest.approx(0.27)


def test_exploration_decays_after_each_episode_down_to_the_minimum():
    agent = QLearningAgent(exploration_rate=1.0, exploration_decay=0.5, min_exploration=0.1)
    agent.update(S1, HIT, 0, S2, done=False)
    assert agent.epsilon == 1.0  # only finished episodes decay exploration
    for _ in range(10):
        agent.update(S1, HIT, 0, S2, done=True)
    assert agent.epsilon == pytest.approx(0.1)


@pytest.mark.parametrize(
    "state, key",
    [
        ({"player_value": 15, "dealer_showing": 7}, (15, 7)),
        ({"obs": [16, 9]}, (16, 9)),
        ((20, 3), (20, 3)),
        ("unexpected", (0, 0)),
    ],
)
def test_state_key_accepts_the_supported_formats(state, key):
    assert QLearningAgent()._state_to_key(state) == key


def test_greedy_evaluation_picks_the_highest_q_value():
    agent = QLearningAgent()
    agent.q_table[(12, 10)] = np.array([0.4, -0.2])
    action, _ = agent.eval_step(S1)
    assert action == HIT


def test_save_and_load_round_trip(tmp_path):
    agent = QLearningAgent()
    agent.q_table[(12, 10)] = np.array([0.1, 0.9])
    path = tmp_path / "q.pkl"
    agent.save_model(str(path))

    loaded = QLearningAgent()
    assert loaded.load_model(str(path))
    assert list(loaded.q_table[(12, 10)]) == pytest.approx([0.1, 0.9])


def test_short_training_beats_random_play():
    np.random.seed(1)
    env = BlackjackEnvironment()
    agent, history = train_q_learning_with_env(env, num_episodes=5000, evaluation_interval=5000)
    assert history["episodes"] == [5000]
    # Random play wins about 28-29% in this environment; a trained agent should do clearly better
    assert evaluate_agent(agent, env, num_games=3000) > 0.35
