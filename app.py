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

experiment_tab, learning_tab = st.tabs(["Experiment", "Learning"])

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

with experiment_tab:
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

with learning_tab:
    st.subheader("What This Program Demonstrates")
    st.markdown(
        """
This app is a compact reinforcement learning experiment focused on the
**exploration versus exploitation** tradeoff. The agent must repeatedly choose
between several reward sources, called arms, without knowing in advance which
one will produce the best long-term outcome.

Each arm returns rewards from a Gamma distribution. Some arms are consistently
better on average, but the agent only discovers that pattern by trying actions,
observing rewards, and updating its estimates over time.
"""
    )

    concept_col, outcome_col = st.columns(2)
    with concept_col:
        st.markdown(
            """
#### What is being learned

- How uncertainty affects decision-making
- How an agent estimates reward quality from experience
- Why early exploration can improve long-term performance
- How strategy choices change total reward and optimal-action rate
"""
        )

    with outcome_col:
        st.markdown(
            """
#### What the app measures

- Running average reward over time
- Percentage of optimal actions selected
- How often each arm is chosen
- Which strategy adapts faster under the selected settings
"""
        )

    st.subheader("Strategies Compared")
    st.markdown(
        """
**Epsilon-Greedy** is simple and practical: it usually chooses the best-known
arm, but occasionally explores a random option. Increasing epsilon makes the
agent more curious, while lowering it makes the agent more exploitative.

**Upper Confidence Bound (UCB)** takes a more directed approach. It favors arms
that either look promising or have not been tested enough. This gives the agent
a principled way to explore uncertainty instead of exploring randomly.
"""
    )

    st.subheader("Why It Matters")
    st.markdown(
        """
Bandit algorithms are useful whenever a system must make repeated choices while
learning from feedback. The same idea appears in recommendation systems,
online experiments, ad allocation, product optimization, and adaptive user
experiences.

This project turns the theory into an interactive simulation: adjust the
parameters, compare strategies, and see how learning behavior changes in real
time.
"""
    )
