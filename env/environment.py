"""Core simulation environment for tracking deepfake spread."""

from __future__ import annotations

import random
import uuid
from typing import Any

from env.models import Action, Observation
from env.reward import compute_reward
from env.tasks import Task, get_task


class DeepfakeEnv:
	"""OpenEnv-style environment for multi-post deepfake spread tracking.
	
	Processes all posts simultaneously per step, with difficulty-based spread rates.
	Reward is averaged across all posts for proper difficulty differentiation.
	"""

	def __init__(
		self,
		task: str = "easy",
		max_views: int = 10_000,
		seed: int | None = None,
	) -> None:
		"""Initialize environment with task configuration.

		Args:
			task: Task difficulty level ("easy", "medium", "hard").
			max_views: Threshold for viral spread termination.
			seed: Random seed for reproducibility.
		"""
		self.task: Task = get_task(task)
		self.max_views = max_views
		# Adjust max_steps by difficulty: harder tasks have fewer decision steps
		if task == "easy":
			self.max_steps = 50
		elif task == "medium":
			self.max_steps = 30
		else:  # hard
			self.max_steps = 15
		self.rng = random.Random(seed)
		self.posts: list[Observation] = []
		self.done = False
		self.step_count = 0

	def reset(self) -> Observation:
		"""Reset and initialize posts based on task difficulty.

		Returns:
			First post observation (for compatibility).
		"""
		self.done = False
		self.step_count = 0
		self.posts = []

		# Generate initial posts for the task
		for i in range(self.task.num_posts):
			# Inverted difficulty: low difficulty (Easy) = high initial prob (easier to detect)
			# High difficulty (Hard) = low initial prob (harder to detect)
			# Easy (0.0): 0.5-0.7, Medium (0.5): 0.3-0.5, Hard (1.0): 0.1-0.3
			prob_min = max(0.1, 0.7 - self.task.detection_difficulty * 0.6)
			prob_max = min(1.0, 0.7 - self.task.detection_difficulty * 0.4)
			
			post = Observation(
				post_id=f"post_{uuid.uuid4().hex[:8]}",
				platform=self.rng.choice(["twitter", "instagram", "whatsapp"]),
				views=self.rng.randint(50, 300),
				shares=self.rng.randint(5, 30),
				deepfake_probability=round(
					self.rng.uniform(prob_min, prob_max),
					3,
				),
				trust_score=round(self.rng.uniform(0.3, 0.8), 3),
				is_flagged=False,
			)
			self.posts.append(post)

		return self.posts[0] if self.posts else Observation(
			post_id="empty", platform="twitter", views=0, shares=0,
			deepfake_probability=0.0, trust_score=0.0, is_flagged=False
		)

	def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
		"""Execute one step: simulate spread on ALL posts, apply action, compute targeted reward.

		Args:
			action: Agent action targeting a specific post.

		Returns:
			Tuple: (observation, reward_for_action, done, info).
		"""
		if self.done:
			raise RuntimeError("Environment is done. Call reset() before step().")

		if not self.posts:
			raise RuntimeError("Environment has no posts. Call reset() first.")

		self.step_count += 1

		# Find target post for the action
		target_post = None
		target_index = -1
		for i, post in enumerate(self.posts):
			if post.post_id == action.post_id:
				target_post = post
				target_index = i
				break

		if target_post is None:
			raise ValueError(f"Unknown post_id: {action.post_id}")

		# Simulate spread and compute rewards for ALL posts
		viral_count = 0
		target_reward = 0.0

		for i, post in enumerate(self.posts):
			# Difficulty-based spread simulation
			spread_increase = self.rng.randint(
				self.task.spread_rate_min,
				self.task.spread_rate_max,
			)
			updated_views = post.views + spread_increase
			updated_shares = post.shares + self.rng.randint(5, 30)

			# Deepfake probability increases over time (proportional to detection difficulty)
			prob_increase = self.rng.uniform(0.01, 0.05 * (1 + self.task.detection_difficulty))
			updated_probability = min(
				1.0,
				round(post.deepfake_probability + prob_increase, 3),
			)

			# Erosion of trust over time
			updated_trust = max(
				0.0,
				round(post.trust_score - self.rng.uniform(0.01, 0.05), 3),
			)

			is_flagged = post.is_flagged

			# Apply action only to the targeted post
			if i == target_index:
				if action.action == "FLAG_POST":
					is_flagged = True
				elif action.action == "REDUCE_VISIBILITY":
					updated_views = max(0, int(updated_views * 0.7))

			# Create updated observation
			updated_post = Observation(
				post_id=post.post_id,
				platform=post.platform,
				views=updated_views,
				shares=updated_shares,
				deepfake_probability=updated_probability,
				trust_score=updated_trust,
				is_flagged=is_flagged,
			)

			# Update post in list
			self.posts[i] = updated_post

			# Compute reward only for the targeted post's action
			if i == target_index:
				early_detection = updated_views < 5000
				target_reward = compute_reward(updated_post, action, early_detection)

			# Count viral posts (without action)
			if updated_views > self.max_views:
				viral_count += 1

		# Termination conditions
		max_steps_reached = self.step_count >= self.max_steps
		majority_viral = viral_count > len(self.posts) * 0.5
		self.done = max_steps_reached or majority_viral

		# Return updated target post for observation
		observation = self.posts[target_index]

		info: dict[str, Any] = {
			"task": self.task.name,
			"max_views": self.max_views,
			"max_steps": self.max_steps,
			"step_count": self.step_count,
			"viral_posts": viral_count,
			"total_posts": len(self.posts),
			"avg_views": round(sum(p.views for p in self.posts) / len(self.posts), 1),
			"avg_deepfake_prob": round(sum(p.deepfake_probability for p in self.posts) / len(self.posts), 3),
		}

		return observation, target_reward, self.done, info

	def state(self) -> dict[str, Any]:
		"""Return full environment state snapshot.

		Returns:
			Serializable state dictionary.
		"""
		viral_count = sum(1 for post in self.posts if post.views > self.max_views)
		return {
			"task": self.task.name,
			"done": self.done,
			"max_views": self.max_views,
			"max_steps": self.max_steps,
			"step_count": self.step_count,
			"total_posts": len(self.posts),
			"viral_posts": viral_count,
			"posts": [post.model_dump() for post in self.posts],
		}
