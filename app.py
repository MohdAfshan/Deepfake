"""Streamlit dashboard for Deepfake Spread Tracker OpenEnv."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from baseline import select_best_action
from env.environment import DeepfakeEnv
from env.models import Action


TASK_LABELS = ["Easy", "Medium", "Hard"]
TASK_MAP = {label: label.lower() for label in TASK_LABELS}
DEFAULT_SEED = 42


st.set_page_config(
    page_title="Deepfake Spread Tracker Dashboard",
    page_icon="📊",
    layout="wide",
)


def get_task_key() -> str:
    """Return the current task key from the selector."""
    return TASK_MAP[st.session_state.selected_task]


def build_environment(task_key: str) -> DeepfakeEnv:
    """Create a fresh environment for the selected task."""
    env = DeepfakeEnv(task=task_key, seed=DEFAULT_SEED)
    env.reset()
    return env


def initialize_session(task_key: str) -> None:
    """Initialize or replace the environment in session state."""
    st.session_state.env = build_environment(task_key)
    st.session_state.task_key = task_key
    st.session_state.previous_shares = {
        post.post_id: post.shares for post in st.session_state.env.posts
    }
    st.session_state.history = [snapshot_metrics(st.session_state.env)]
    st.session_state.last_reward = 0.0
    st.session_state.last_action = ""
    st.session_state.last_target = ""


def snapshot_metrics(env: DeepfakeEnv) -> dict[str, float | int]:
    """Capture the current aggregate environment metrics."""
    total_posts = len(env.posts)
    viral_posts = sum(1 for post in env.posts if post.views > env.max_views)
    average_prob = (
        sum(post.deepfake_probability for post in env.posts) / total_posts
        if total_posts
        else 0.0
    )
    average_views = (
        sum(post.views for post in env.posts) / total_posts if total_posts else 0.0
    )
    return {
        "step": env.step_count,
        "avg_views": round(average_views, 2),
        "avg_deepfake_probability": round(average_prob, 3),
        "viral_posts": viral_posts,
        "total_posts": total_posts,
    }


def choose_action(env: DeepfakeEnv) -> Action:
    """Select the highest-priority baseline action for the current state."""
    previous_shares = st.session_state.previous_shares
    best_post = None
    best_action = "DO_NOTHING"
    best_priority = -1
    best_probability = -1.0

    for post in env.posts:
        action_name, priority = select_best_action(
            post,
            previous_shares.get(post.post_id),
        )
        if priority > best_priority or (
            priority == best_priority and post.deepfake_probability > best_probability
        ):
            best_post = post
            best_action = action_name
            best_priority = priority
            best_probability = post.deepfake_probability

    if best_post is None:
        best_post = env.posts[0]

    return Action(action=best_action, post_id=best_post.post_id)


def run_one_step() -> None:
    """Advance the environment by one baseline-driven step."""
    env = st.session_state.env
    action = choose_action(env)
    _, reward, done, _ = env.step(action)

    st.session_state.previous_shares = {
        post.post_id: post.shares for post in env.posts
    }
    st.session_state.last_reward = reward
    st.session_state.last_action = action.action
    st.session_state.last_target = action.post_id
    st.session_state.history.append(snapshot_metrics(env))
    st.session_state.done = done


def run_full_simulation() -> None:
    """Run the baseline policy until the environment terminates."""
    env = st.session_state.env
    safety_limit = max(env.max_steps, 1) * 2
    iterations = 0

    while not env.done and iterations < safety_limit:
        action = choose_action(env)
        _, reward, done, _ = env.step(action)
        st.session_state.previous_shares = {
            post.post_id: post.shares for post in env.posts
        }
        st.session_state.last_reward = reward
        st.session_state.last_action = action.action
        st.session_state.last_target = action.post_id
        st.session_state.history.append(snapshot_metrics(env))
        st.session_state.done = done
        iterations += 1


def reset_environment() -> None:
    """Reset the environment for the current task selection."""
    initialize_session(get_task_key())


def render_posts_table(env: DeepfakeEnv) -> None:
    """Render the current posts with visual highlighting."""
    frame = pd.DataFrame(
        [
            {
                "post_id": post.post_id,
                "platform": post.platform,
                "views": post.views,
                "shares": post.shares,
                "deepfake_probability": post.deepfake_probability,
                "trust_score": post.trust_score,
                "is_flagged": post.is_flagged,
            }
            for post in env.posts
        ]
    )

    if frame.empty:
        st.info("No posts available. Reset the environment.")
        return

    column_index = {name: idx for idx, name in enumerate(frame.columns)}

    def style_rows(row: pd.Series) -> list[str]:
        styles = ["" for _ in row.index]
        if bool(row["is_flagged"]):
            return ["background-color: #fff2a8; color: #3d3200;" for _ in row.index]
        if int(row["views"]) > env.max_views:
            return ["background-color: #ffd08a; color: #5c2f00;" for _ in row.index]
        if float(row["deepfake_probability"]) > 0.7:
            styles[column_index["deepfake_probability"]] = (
                "background-color: #ffb3b3; color: #7a0000; font-weight: 700;"
            )
        return styles

    styled = frame.style.apply(style_rows, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_metrics(env: DeepfakeEnv) -> None:
    """Render aggregate metrics and chart data."""
    metrics = snapshot_metrics(env)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total posts", metrics["total_posts"])
    metric_cols[1].metric("Viral posts", metrics["viral_posts"])
    metric_cols[2].metric("Average deepfake probability", f"{metrics['avg_deepfake_probability']:.2f}")
    metric_cols[3].metric("Step count", metrics["step"])

    history_frame = pd.DataFrame(st.session_state.history)
    if not history_frame.empty:
        chart_frame = history_frame.set_index("step")[["avg_views", "avg_deepfake_probability"]]
        st.line_chart(chart_frame, use_container_width=True)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("Deepfake Spread Tracker Dashboard")
    st.write("Monitor deepfake spread, apply containment actions, and track simulation metrics across difficulty levels.")

    if "selected_task" not in st.session_state:
        st.session_state.selected_task = "Easy"

    selected_task = st.selectbox("Task Selector", TASK_LABELS, key="selected_task")
    task_key = TASK_MAP[selected_task]

    if "env" not in st.session_state or st.session_state.get("task_key") != task_key:
        initialize_session(task_key)

    env = st.session_state.env

    button_cols = st.columns(3)
    if button_cols[0].button("Reset Environment", use_container_width=True):
        reset_environment()
        env = st.session_state.env

    if button_cols[1].button("Run One Step", use_container_width=True, disabled=env.done):
        run_one_step()
        env = st.session_state.env

    if button_cols[2].button("Run Full Simulation", use_container_width=True, disabled=env.done):
        run_full_simulation()
        env = st.session_state.env

    if env.done:
        st.success("Simulation complete.")

    render_metrics(env)

    st.subheader("Posts")
    render_posts_table(env)

    if st.session_state.last_action:
        st.caption(
            f"Last action: {st.session_state.last_action} on {st.session_state.last_target} | "
            f"Last reward: {st.session_state.last_reward:.2f}"
        )


if __name__ == "__main__":
    main()