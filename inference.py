from env.environment import DeepfakeEnv
from env.models import Action

env = None


def reset(task="easy", seed=42):
    global env
    env = DeepfakeEnv(task=task, seed=seed)
    obs = env.reset()
    return obs


def step(action_dict):
    global env

    action = Action(
        action=action_dict["action"],
        post_id=action_dict["post_id"]
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }
