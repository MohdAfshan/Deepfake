"""Task definitions for the Deepfake OpenEnv environment."""

from dataclasses import dataclass


@dataclass
class Task:
	"""Configuration for a difficulty level."""

	name: str
	num_posts: int
	spread_rate_min: int
	spread_rate_max: int
	detection_difficulty: float  # 0.0 = easy, 1.0 = hard


# Easy task: 5 posts, slow spread (100-200), obvious deepfakes
EASY = Task(
	name="easy",
	num_posts=5,
	spread_rate_min=100,
	spread_rate_max=200,
	detection_difficulty=0.0,  # Easy detection
)

# Medium task: 20 posts, moderate spread (300-600), moderate detection
MEDIUM = Task(
	name="medium",
	num_posts=20,
	spread_rate_min=300,
	spread_rate_max=600,
	detection_difficulty=0.5,
)

# Hard task: 100 posts, fast spread (600-1500), subtle deepfakes
HARD = Task(
	name="hard",
	num_posts=100,
	spread_rate_min=600,
	spread_rate_max=1500,
	detection_difficulty=1.0,  # Hard detection
)

TASKS = {"easy": EASY, "medium": MEDIUM, "hard": HARD}


def get_task(name: str) -> Task:
	"""Retrieve task configuration by name."""
	if name not in TASKS:
		raise ValueError(f"Unknown task: {name}. Choose from {list(TASKS.keys())}")
	return TASKS[name]
