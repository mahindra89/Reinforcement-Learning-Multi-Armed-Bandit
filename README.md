---
title: Bandit Strategy Lab
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
---

# Bandit Strategy Lab

An interactive reinforcement learning demo for studying exploration versus
exploitation in stochastic reward environments.

The app compares **epsilon-greedy** and **Upper Confidence Bound (UCB)** on a
multi-armed bandit problem where each arm pays rewards from a Gamma
distribution. It turns a static experiment into a small decision-systems lab:
change the number of steps, exploration rate, UCB confidence bonus, random
seed, and reward setup, then watch how the strategies behave.

## Live Demo

This repository is ready to deploy on **Hugging Face Spaces** using the
Streamlit SDK. Create a new Space, choose Streamlit, and point it at this repo.

## Why This Project Matters

Multi-armed bandits are a practical model for online decision-making problems:

- A/B test allocation
- recommendation systems
- ad ranking
- experiment design
- adaptive product optimization

The project focuses on the core tradeoff: spend decisions learning about
uncertain options, or exploit the option that currently looks best.

## Features

- Interactive Streamlit interface
- Configurable decision horizon, random seed, and strategy parameters
- Original five-arm Gamma reward setup
- Generated arm setup for quick what-if experiments
- Running average reward chart
- Optimal-action selection chart
- Arm pull-count comparison
- Reusable simulation code in `bandit_lab.py`

## Strategies Compared

### Epsilon-Greedy

With probability `epsilon`, the agent explores by selecting a random arm.
Otherwise, it exploits the arm with the highest estimated reward.

### Upper Confidence Bound

UCB adds an uncertainty bonus to each estimated reward:

```text
Q(a) + c * sqrt(log(t) / N(a))
```

This encourages the agent to try arms with less evidence before committing to
the strongest option.

## Project Structure

```text
.
├── app.py          # Streamlit interface
├── bandit_lab.py   # Reusable simulation logic
├── rla1-2.py       # Original experiment script
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy On Hugging Face Spaces

1. Create a Hugging Face account.
2. Create a new Space.
3. Select **Streamlit** as the SDK.
4. Connect or upload this GitHub repository.
5. Hugging Face will use `app.py` and `requirements.txt` to build the app.

The YAML block at the top of this README is included so Hugging Face Spaces can
detect the app configuration.

## Next Improvements

- Add Thompson Sampling as a third strategy.
- Add downloadable experiment results.
- Add preset scenarios for high-variance and close-mean reward arms.
- Add unit tests for reward sampling and strategy behavior.
