from env.environment import DeepfakeEnv
from env.models import Action
import sys

class Agent:

    def __init__(self):
        self.env = None

    def reset(self, config=None):
        task_name = "reset_env"
        print(f"[START] task={task_name}", flush=True)

        task = "easy"
        seed = 42

        if config:
            task = config.get("task", "easy")
            seed = config.get("seed", 42)

        self.env = DeepfakeEnv(task=task, seed=seed)
        observation = self.env.reset()

        # Log first step (optional, reward 0 for reset)
        print(f"[STEP] step=1 reward=0.0", flush=True)

        # End of reset task
        print(f"[END] task={task_name} score=0.0 steps=1", flush=True)

        return observation

    def step(self, action):
        task_name = "step_env"
        print(f"[START] task={task_name}", flush=True)

        action_obj = Action(
            action=action["action"],
            post_id=action["post_id"]
        )

        observation, reward, done, info = self.env.step(action_obj)

        # Log this step
        print(f"[STEP] step=1 reward={reward}", flush=True)

        # End of step task
        print(f"[END] task={task_name} score={reward} steps=1", flush=True)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }


# Initialize agent globally
agent = Agent()


def reset(config=None):
    return agent.reset(config)


def step(action):
    return agent.step(action)


# Optional: allow running inference.py standalone for testing
if __name__ == "__main__":
    # Test reset
    obs = reset()
    print("Reset observation:", obs, flush=True)

    # Test step with dummy action
    test_action = {"action": 0, "post_id": 0}
    step_result = step(test_action)
    print("Step result:", step_result, flush=True)
