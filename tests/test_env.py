from app.env import GDMultiEvalEnv
from app.models import GDAction


def test_easy_env_runs():
    env = GDMultiEvalEnv(difficulty="easy")
    obs = env.reset()
    assert obs.topic is not None

    obs, reward, done, info = env.step(GDAction(action_type="submit_final_evaluation"))
    assert done is True


def test_medium_env_runs():
    env = GDMultiEvalEnv(difficulty="medium")
    obs = env.reset()
    assert obs.topic is not None

    obs, reward, done, info = env.step(GDAction(action_type="request_more_context"))
    assert reward.value is not None


def test_hard_env_runs():
    env = GDMultiEvalEnv(difficulty="hard")
    obs = env.reset()
    assert obs.topic is not None

    obs, reward, done, info = env.step(GDAction(action_type="flag_dominance", speaker="Karthik"))
    obs, reward, done, info = env.step(GDAction(action_type="flag_irrelevant_speaker", speaker="Arun"))
    obs, reward, done, info = env.step(GDAction(action_type="flag_low_participation", speaker="Arun"))
    obs, reward, done, info = env.step(GDAction(action_type="select_winner", speaker="Divya"))
    obs, reward, done, info = env.step(GDAction(action_type="submit_final_evaluation"))

    assert done is True
    assert reward.value >= 0.5