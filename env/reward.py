"""Reward computation logic for the Deepfake OpenEnv environment."""

from env.models import Action, Observation


def compute_reward(
    observation: Observation,
    action: Action,
    early_detection: bool = False,
) -> float:
    """
    Compute normalized reward [0.0, 1.0] based on action quality.

    Args:
        observation: Current post state after action.
        action: Action taken by the agent.
        early_detection: Whether the post was flagged before viral spread.

    Returns:
        Normalized reward in range [0.0, 1.0].
    """
    reward = 0.0

    # Core deepfake flagging reward
    if action.action == "FLAG_POST":
        if observation.deepfake_probability > 0.5:
            # Correct detection: +0.5
            reward += 0.5
            # Bonus for early detection (views < 5000)
            if observation.views < 5000:
                reward += 0.3
        else:
            # False flag: -0.2
            reward -= 0.2

    # Reward for reducing visibility on suspected deepfakes
    elif action.action == "REDUCE_VISIBILITY":
        if observation.deepfake_probability > 0.5 and observation.views < 10000:
            reward += 0.2

    # Reward for tracing source of suspected content
    elif action.action == "TRACE_SOURCE":
        if observation.deepfake_probability > 0.6:
            reward += 0.1

    # Penalty for inaction when content is likely deepfake
    elif action.action == "DO_NOTHING":
        if observation.deepfake_probability > 0.7:
            reward -= 0.2

    # Large penalty for viral spread (views > 10000)
    if observation.views > 10000:
        reward -= 0.4
        # Partial recovery if flagged after viral
        if observation.is_flagged:
            reward += 0.1

    # Normalize reward to [0.0, 1.0]
    normalized_reward = max(0.0, min(1.0, reward))

    return round(normalized_reward, 3)
