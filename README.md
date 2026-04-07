# Deepfake Spread Tracker — OpenEnv Environment

A realistic multi-agent simulation environment for detecting and containing deepfake video spread across social media platforms.

## Overview

### The Problem

Deepfake videos pose a significant threat to information integrity across social media. These synthetic videos can spread rapidly, harm public trust, and cause substantial damage before detection. The challenge is compounded by:

- **Rapid viral spread**: Posts can reach millions within hours
- **Detection difficulty**: Modern deepfakes are increasingly subtle and hard to identify
- **Scale**: Monitoring millions of posts simultaneously is computationally challenging
- **Platform diversity**: Deepfakes spread across multiple platforms (Twitter, Instagram, WhatsApp, etc.)

### The Solution

This OpenEnv environment simulates realistic deepfake spread dynamics, allowing AI agents to develop and test detection and containment strategies. Agents must:

1. **Detect** deepfake content by analyzing post characteristics
2. **Alert** platforms to FLAG or REDUCE VISIBILITY of suspicious content
3. **Trace** sources to understand propagation patterns
4. **Minimize** viral spread and misinformation impact

### Multi-Task Learning

The environment provides three difficulty levels for progressive skill development:

- **Easy**: 5 obvious deepfakes, slow spread → agents develop detection fundamentals
- **Medium**: 20 moderate-difficulty posts, diverse platforms → realistic complexity
- **Hard**: 100 subtle deepfakes, rapid viral spread → production-level performance

---

## Features

✓ **Real-World Simulation**
- Simulates authentic social media dynamics (views, shares, platform diversity)
- Posts evolve over time with increasing deepfake probability and trust erosion
- Stochastic spread rates based on task difficulty

✓ **3 Difficulty Levels**
- Easy (5 posts, obvious deepfakes, 100-200 views/step)
- Medium (20 posts, moderate difficulty, 300-600 views/step)
- Hard (100 posts, subtle deepfakes, 600-1500 views/step)

✓ **Intelligent Reward Shaping**
- Early detection bonuses (+0.3 for <5K views)
- Correct flagging rewards (+0.5 for accurate detection)
- Viral spread penalties (-0.4 for >10K views)
- False flag penalties (-0.2 for incorrect detections)
- Normalized rewards in [0.0, 1.0] range

✓ **Baseline Agent**
- Rule-based baseline achieving 0.70 (Easy), 0.53 (Medium), 0.24 (Hard)
- Demonstrates intelligent decision-making vs. random policy
- Prioritizes high-confidence actions

✓ **OpenEnv Compliance**
- Standard `reset()`, `step()`, `state()` interface
- Pydantic models for type safety
- Full simulator control and state visibility

---

## Environment Design

### Observation Space

Each post observation includes:

```python
class Observation(BaseModel):
    post_id: str                          # Unique post identifier
    platform: str                         # "twitter" | "instagram" | "whatsapp"
    views: int                            # Current view count (≥0)
    shares: int                           # Current share count (≥0)
    deepfake_probability: float           # Likelihood of being deepfake [0.0, 1.0]
    trust_score: float                    # Community trust level [0.0, 1.0]
    is_flagged: bool                      # Whether post has been flagged
```

### Action Space

Agents choose one of four actions per step:

```python
class Action(BaseModel):
    action: str                           # "FLAG_POST" | "REDUCE_VISIBILITY" | "TRACE_SOURCE" | "DO_NOTHING"
    post_id: str                          # Target post identifier
```

**Action Descriptions:**

- **FLAG_POST**: Permanently flag the post as deepfake (removes from circulation)
- **REDUCE_VISIBILITY**: Reduce post visibility by 30% (slow spread)
- **TRACE_SOURCE**: Initiate source investigation (gather intelligence)
- **DO_NOTHING**: Take no action (post continues spreading naturally)

### Reward Signal

Rewards are normalized to [0.0, 1.0] and computed as:

```python
+0.5  for correct deepfake detection (FLAG_POST when probability > 0.5)
+0.3  for early detection (views < 5000)
+0.2  for reducing visibility of suspected deepfakes
-0.2  for false flags or inaction on obvious deepfakes
-0.4  for viral spread (views > 10000)
```

Final reward: `clamp(reward, 0.0, 1.0)`

---

## Tasks

### Easy Task

**Configuration:**
- **Posts**: 5 initial posts
- **Spread Rate**: 100–200 views per step
- **Initial Deepfake Probability**: 0.5–0.7 (obvious)
- **Detection Difficulty**: 0.0 (easiest)
- **Max Steps**: 50

**Characteristics:**
- Manageable number of posts
- Slow, predictable spread
- Clear deepfake signals
- Ideal for baseline testing

### Medium Task

**Configuration:**
- **Posts**: 20 initial posts
- **Spread Rate**: 300–600 views per step
- **Initial Deepfake Probability**: 0.3–0.5 (moderate)
- **Detection Difficulty**: 0.5 (moderate)
- **Max Steps**: 30

**Characteristics:**
- Realistic post volume
- Multi-platform spread (Twitter, Instagram, WhatsApp)
- Moderate detection signals
- Suitable for agent development

### Hard Task

**Configuration:**
- **Posts**: 100 initial posts
- **Spread Rate**: 600–1500 views per step
- **Initial Deepfake Probability**: 0.1–0.3 (subtle)
- **Detection Difficulty**: 1.0 (hardest)
- **Max Steps**: 15

**Characteristics:**
- Large-scale post monitoring
- Rapid viral spread (each post can gain 1.5K views per step)
- Subtle deepfake signals (harder to detect)
- Requires high-performance agents

---

## Reward Function

### Computation Logic

The reward function balances multiple objectives:

```python
def compute_reward(observation, action, early_detection):
    reward = 0.0
    
    # Core: Correct detection
    if action == "FLAG_POST":
        if observation.deepfake_probability > 0.5:
            reward += 0.5                           # Correct flag
            if early_detection (views < 5000):
                reward += 0.3                       # Early detection bonus
        else:
            reward -= 0.2                           # False flag penalty
    
    # Support: Reduce visibility / trace source
    elif action == "REDUCE_VISIBILITY":
        if observation.deepfake_probability > 0.5:
            reward += 0.2
    
    elif action == "TRACE_SOURCE":
        if observation.deepfake_probability > 0.6:
            reward += 0.1
    
    # Penalty: Inaction on obvious deepfakes
    elif action == "DO_NOTHING":
        if observation.deepfake_probability > 0.7:
            reward -= 0.2
    
    # Large penalty: Viral spread
    if observation.views > 10000:
        reward -= 0.4
        if observation.is_flagged:
            reward += 0.1                           # Partial recovery if flagged
    
    return max(0.0, min(1.0, reward))               # Normalize
```

### Reward Incentives

- **Early Detection**: Agents are rewarded for catching deepfakes before they go viral
- **Accuracy**: Correct flags are rewarded; false flags are penalized
- **Proactivity**: Reducing visibility and tracing sources provide partial rewards
- **Learning**: Inaction on obvious deepfakes is penalized

---

## Baseline Results

A rule-based baseline agent using priority-based decision logic achieves:

```
Running Deepfake OpenEnv Rule-Based Baseline...

Task Easy Score: 0.70
Task Medium Score: 0.53
Task Hard Score: 0.24

Average Score: 0.49
```

### Baseline Strategy

The rule-based baseline uses the following priority logic:

1. **Priority 1 (FLAG_POST)**: If deepfake_probability > 0.6
2. **Priority 2 (TRACE_SOURCE)**: If views > 5000 AND deepfake_probability > 0.4
3. **Priority 3 (REDUCE_VISIBILITY)**: If shares increase > 100 in one step
4. **Priority 4 (DO_NOTHING)**: Default fallback

The agent evaluates all posts each step and selects the highest-priority action (or highest-confidence post for ties).

### Performance Analysis

- **Easy (0.70)**: High score due to obvious signals and manageable scale
- **Medium (0.53)**: Moderate score reflects increased post volume and subtlety
- **Hard (0.24)**: Low score reflects extreme difficulty: 100 posts, rapid spread, subtle signals

---

## Setup Instructions

### Requirements

- Python 3.8+
- pip

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd deepfake-openenv
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Baseline

Execute the rule-based baseline:

```bash
python baseline.py
```

### HuggingFace Spaces

The project now uses a Streamlit dashboard entrypoint:

```bash
python -m streamlit run app.py
```

The dashboard connects directly to `DeepfakeEnv`, lets you switch between Easy, Medium, and Hard, and provides buttons to reset the environment, run a single step, or run a full simulation.

### Deployment Notes

- Install dependencies from `requirements.txt`
- Keep the container lightweight with the provided `python:3.10-slim` image
- Run `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=7860` in Spaces or Docker

Expected output:
```
Running Deepfake OpenEnv Rule-Based Baseline...

Task Easy Score: 0.70
Task Medium Score: 0.53
Task Hard Score: 0.24

Average Score: 0.49
```

### Running a Single Task

Test your agent on a specific task:

```python
from env.environment import DeepfakeEnv
from env.models import Action

# Initialize environment
env = DeepfakeEnv(task="easy", seed=42)
observation = env.reset()

# Run one step
action = Action(action="FLAG_POST", post_id=observation.post_id)
observation, reward, done, info = env.step(action)

print(f"Reward: {reward}, Done: {done}")
```

---

## Project Structure

```
deepfake-openenv/
│
├── env/
│   ├── __pycache__/
│   ├── models.py           # Pydantic models for Observation, Action, Reward
│   ├── tasks.py            # Task definitions (Easy, Medium, Hard)
│   ├── reward.py           # Reward computation function
│   └── environment.py      # DeepfakeEnv core simulation
│
├── baseline.py             # Rule-based baseline agent
├── app.py                  # HuggingFace Spaces entrypoint
├── openenv.yaml            # OpenEnv configuration and metadata
├── Dockerfile              # Production container setup
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
└── venv/                   # Virtual environment (local)
```

### File Descriptions

- **env/models.py**: Defines Observation, Action, and Reward Pydantic models with validation
- **env/tasks.py**: Task configurations (post count, spread rates, difficulty levels)
- **env/reward.py**: `compute_reward()` function with rule-based reward logic
- **env/environment.py**: `DeepfakeEnv` class implementing OpenEnv interface
- **baseline.py**: `run_episode()` and priority-based decision logic for baseline agent
- **app.py**: Streamlit dashboard for the Deepfake Spread Tracker simulation
- **openenv.yaml**: Metadata and task descriptions for OpenEnv hackathon submission
- **Dockerfile**: Container setup for deployment (`streamlit run app.py`)

---

## HuggingFace Deployment

### How to run

Use the app entrypoint:

```bash
python app.py
```

If you are building the container locally, the Dockerfile installs `requirements.txt` and launches the same entrypoint automatically.

### Environment description

Deepfake Spread Tracker is a simulation environment for content moderation and misinformation containment. Each step advances all posts simultaneously, and the agent decides whether to flag, reduce visibility, trace the source, or do nothing.

The environment uses normalized rewards in the `[0.0, 1.0]` range and exposes the standard `reset()`, `step()`, and `state()` interface.

### Tasks

- **Easy**: 5 posts, slow spread, obvious deepfake signals
- **Medium**: 20 posts, moderate spread, mixed signals
- **Hard**: 100 posts, rapid spread, subtle deepfake signals

The expected baseline scores are:

- Easy: 0.70
- Medium: 0.53
- Hard: 0.24

---

## OpenEnv Compliance

This environment fully complies with the OpenEnv standard interface:

### `reset() → Observation`

Initializes a new episode:
- Generates posts based on task difficulty
- Resets step counter and done flag
- Returns first post observation

```python
env = DeepfakeEnv(task="easy")
observation = env.reset()
```

### `step(action: Action) → (Observation, float, bool, dict)`

Advances the simulation by one step:
- Simulates spread on ALL posts
- Applies action to targeted post
- Computes reward
- Returns tuple: (observation, reward, done, info)

```python
action = Action(action="FLAG_POST", post_id=observation.post_id)
observation, reward, done, info = env.step(action)
```

**Returns:**
- `observation`: Updated post state (Observation model)
- `reward`: Normalized reward in [0.0, 1.0]
- `done`: Episode termination flag (bool)
- `info`: Metadata dict with task, step count, viral count, etc.

### `state() → dict`

Returns full environment state snapshot:
- Post list with all current states
- Task configuration
- Step count and done status

```python
state = env.state()
print(f"Posts: {len(state['posts'])}, Done: {state['done']}")
```

### Key Features

✓ **Deterministic**: Fixed seed produces reproducible results
✓ **Vectorizable**: All posts simulated simultaneously per step
✓ **Observable**: Full state visibility for analysis and debugging
✓ **Scalable**: Handles 100 posts per step efficiently

---

## Citation

If you use this environment in research, please cite:

```bibtex
@software{deepfake_openenv_2026,
  title={Deepfake Spread Tracker — OpenEnv Environment},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/deepfake-openenv}
}
```

---

## License

MIT License — See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your improvements

---

## Contact

For questions, issues, or feedback, please open an issue on GitHub or contact the maintainers.

---

**Last Updated**: April 3, 2026  
**OpenEnv Version**: 1.0  
**Python Version**: 3.8+