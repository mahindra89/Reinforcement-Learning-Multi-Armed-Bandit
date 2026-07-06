import pandas as pd
import streamlit as st

from bandit_lab import (
    DEFAULT_ARMS,
    arm_table,
    compare_strategies,
    make_progressive_arms,
)


st.set_page_config(
    page_title="Bandit Strategy Lab",
    layout="wide",
)

st.title("Bandit Strategy Lab")
st.caption(
    "Explore how epsilon-greedy and UCB balance exploration and exploitation "
    "when rewards come from Gamma-distributed arms."
)

with st.sidebar:
    st.header("Simulation")
    preset = st.radio(
        "Arm setup",
        ["Original 5-arm setup", "Generated arms"],
        index=0,
    )
    steps = st.slider("Decision steps", 100, 5000, 1000, step=100)
    epsilon = st.slider("Epsilon-greedy exploration", 0.0, 0.5, 0.1, step=0.01)
    confidence = st.slider("UCB confidence bonus", 0.5, 4.0, 2.0, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)

    if preset == "Generated arms":
        arm_count = st.slider("Number of arms", 3, 10, 5)
        spread = st.slider("Reward spread", 0.0, 5.0, 1.5, step=0.1)
        arms = make_progressive_arms(arm_count, spread)
    else:
        arms = DEFAULT_ARMS

results = compare_strategies(
    arms=arms,
    steps=steps,
    epsilon=epsilon,
    confidence=confidence,
    seed=int(seed),
)

best_arm = max(arms, key=lambda arm: arm.mean)
col1, col2, col3 = st.columns(3)
col1.metric("Best theoretical arm", best_arm.name)
col2.metric("Best expected reward", f"{best_arm.mean:.2f}")
col3.metric("Arms compared", len(arms))

st.subheader("Reward Model")
st.dataframe(pd.DataFrame(arm_table(arms)), use_container_width=True)

avg_reward = pd.DataFrame(
    {
        name: run["running_average"]
        for name, run in results.items()
    }
)
optimal_rate = pd.DataFrame(
    {
        name: run["optimal_rate"]
        for name, run in results.items()
    }
)

left, right = st.columns(2)
with left:
    st.subheader("Average Reward Over Time")
    st.line_chart(avg_reward)

with right:
    st.subheader("Optimal Action Selection")
    st.line_chart(optimal_rate)

summary_rows = []
for name, run in results.items():
    pulls = run["pulls"]
    selected_arm = arms[int(pulls.argmax())]
    summary_rows.append(
        {
            "Strategy": name,
            "Total reward": round(run["total_reward"], 2),
            "Final avg reward": round(float(run["running_average"][-1]), 3),
            "Final optimal action %": round(float(run["optimal_rate"][-1]), 2),
            "Most selected arm": selected_arm.name,
        }
    )

st.subheader("Strategy Summary")
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.subheader("Arm Pull Counts")
pull_counts = pd.DataFrame(
    {
        name: {
            arms[idx].name: int(count)
            for idx, count in enumerate(run["pulls"])
        }
        for name, run in results.items()
    }
)
st.bar_chart(pull_counts)

st.markdown(
    """
### What to look for

UCB usually explores each arm early, then concentrates decisions on the arm
with the strongest reward signal. Epsilon-greedy keeps exploring at the rate
you choose, which can help when estimates are noisy but can also reduce total
reward after the best arm is known.
"""
)
