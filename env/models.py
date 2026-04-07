"""Pydantic models for the Deepfake OpenEnv environment."""

from typing import Literal

from pydantic import BaseModel, Field


Platform = Literal["twitter", "instagram", "whatsapp"]
ActionType = Literal["FLAG_POST", "REDUCE_VISIBILITY", "TRACE_SOURCE", "DO_NOTHING"]


class Observation(BaseModel):
	"""Observation returned by the environment at each step."""

	post_id: str
	platform: Platform
	views: int = Field(ge=0)
	shares: int = Field(ge=0)
	deepfake_probability: float = Field(ge=0.0, le=1.0)
	trust_score: float = Field(ge=0.0, le=1.0)
	is_flagged: bool


class Action(BaseModel):
	"""Action taken by an agent against a specific post."""

	action: ActionType
	post_id: str


class Reward(BaseModel):
	"""Reward returned by the environment after each step."""

	score: float
