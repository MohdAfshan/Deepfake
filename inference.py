from env.environment import DeepfakeEnv
from env.models import Action

env = None


def reset(config=None):
    global env

    task = "easy"
    seed = 42

    if config:
        task = config.get("task", "easy")
        seed = config.get("seed", 42)

    env = DeepfakeEnv(task=task, seed=seed)
    observation = env.reset()

    return observation


def step(action):
    global env

    action_obj = Action(
        action=action["action"],
        post_id=action["post_id"]
    )

    observation, reward, done, info = env.step(action_obj)

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info
    }
