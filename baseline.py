"""Baseline runner for the Deepfake OpenEnv environment."""

from env.environment import DeepfakeEnv
from env.models import Action


def select_best_action(observation, previous_shares=None):
	"""Select best action based on post state using rule-based logic.

	Rules (in priority order):
	1. If deepfake_probability > 0.6 → FLAG_POST
	2. If views > 5000 and deepfake_probability > 0.4 → TRACE_SOURCE
	3. If shares growing rapidly (>100 increase) → REDUCE_VISIBILITY
	4. Otherwise → DO_NOTHING

	Args:
		observation: Current post observation.
		previous_shares: Previous shares count for this post.

	Returns:
		Tuple of (action_string, priority_score).
	"""
	# Rule 1: High deepfake probability → flag immediately
	if observation.deepfake_probability > 0.6:
		return "FLAG_POST", 4

	# Rule 2: Viral views + moderate deepfake probability → trace source
	if observation.views > 5000 and observation.deepfake_probability > 0.4:
		return "TRACE_SOURCE", 3

	# Rule 3: Shares growing rapidly → reduce visibility
	if previous_shares is not None:
		shares_increase = observation.shares - previous_shares
		if shares_increase > 100:
			return "REDUCE_VISIBILITY", 2

	# Rule 4: Default to do nothing
	return "DO_NOTHING", 1


def run_episode(task: str, seed: int | None = None) -> float:
	"""Run a single episode on a task with rule-based agent.

	Agent evaluates all posts, selects best action for each, and prioritizes
	high-priority actions (FLAG > TRACE > REDUCE > DO_NOTHING).

	Args:
		task: Task name ("easy", "medium", "hard").
		seed: Random seed for reproducibility.

	Returns:
		Average reward across the episode.
	"""
	env = DeepfakeEnv(task=task, seed=seed)
	observation = env.reset()
	done = False
	total_reward = 0.0
	step_count = 0
	previous_shares = {post.post_id: post.shares for post in env.posts}

	while not done:
		# Evaluate all posts and select best action
		best_post = None
		best_action_str = None
		best_priority = 0
		best_deepfake_prob = 0.0

		for post in env.posts:
			# Select action for this post
			action_str, priority = select_best_action(
				post, previous_shares.get(post.post_id)
			)

			# Prioritize: highest priority first, then by deepfake probability
			if (
				priority > best_priority
				or (
					priority == best_priority
					and post.deepfake_probability > best_deepfake_prob
				)
			):
				best_post = post
				best_action_str = action_str
				best_priority = priority
				best_deepfake_prob = post.deepfake_probability

		if best_post is None:
			break

		# Update previous shares for this post
		previous_shares[best_post.post_id] = best_post.shares

		# Execute action
		action = Action(action=best_action_str, post_id=best_post.post_id)
		observation, reward, done, info = env.step(action)
		total_reward += reward
		step_count += 1

	# Score is average reward across episode
	score = (total_reward / step_count) if step_count > 0 else 0.0
	return round(score, 2)


def main() -> None:
	"""Run episodes for all tasks with rule-based agent and report results."""
	tasks = ["easy", "medium", "hard"]
	scores = {}

	print("Running Deepfake OpenEnv Rule-Based Baseline...\n")

	for task in tasks:
		score = run_episode(task, seed=42)
		scores[task] = score
		print(f"Task {task.capitalize()} Score: {score:.2f}")

	avg_score = sum(scores.values()) / len(scores)
	print(f"\nAverage Score: {avg_score:.2f}")


if __name__ == "__main__":
	main()
