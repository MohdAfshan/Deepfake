from env.environment import DeepfakeEnv
from env.models import Action


class Agent:

    def __init__(self):
        self.env = None

    def reset(self, config=None):
        task = "easy"
        seed = 42

        if config:
            task = config.get("task", "easy")
            seed = config.get("seed", 42)

        self.env = DeepfakeEnv(task=task, seed=seed)
        observation = self.env.reset()

        return observation

    def step(self, action):

        action_obj = Action(
            action=action["action"],
            post_id=action["post_id"]
        )

        observation, reward, done, info = self.env.step(action_obj)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }


agent = Agent()


def reset(config=None):
    return agent.reset(config)


def step(action):
    return agent.step(action)
