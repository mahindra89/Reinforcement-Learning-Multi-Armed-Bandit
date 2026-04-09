# Reinforcement Learning Homework 1 - Multi-Armed Bandit

## Overview
This project implements a **multi-armed bandit** experiment for a Reinforcement Learning homework assignment. The goal is to:

1. Design a bandit environment using **Gamma-distributed reward arms**.
2. Visualize the reward distributions of the arms.
3. Identify the arm with the highest expected reward.
4. Compare two action-selection strategies:
   - **Epsilon-Greedy**
   - **Upper Confidence Bound (UCB)**
5. Evaluate the strategies using:
   - **Average reward over time**
   - **Percentage of optimal action selections over time**

The README is based directly on the uploaded Python script.

---

## File in this Project
- `rla1-2.py` - Main Python script containing all tasks for the homework.

---

## Objective
The script studies how an agent learns to choose the best arm in a stochastic bandit problem. Each arm gives rewards drawn from a Gamma distribution, so the rewards are random but follow a known pattern.

The project is divided into two main tasks:

### Task 1: Designing the Bandit
- Define 5 different arms using Gamma distributions.
- Plot sampled reward histograms and theoretical probability density functions.
- Compute the expected mean reward of each arm.
- Identify the optimal arm.

### Task 2: Training the Agent
- Train an agent using epsilon-greedy and UCB.
- Compare how fast each method learns.
- Plot performance metrics over time.

---

## Bandit Setup
The code defines the following five arms:

| Arm | Shape | Scale | Mean = Shape x Scale |
|-----|-------|-------|----------------------|
| Arm 1 | 0.7 | 2.0 | 1.40 |
| Arm 2 | 1.5 | 2.0 | 3.00 |
| Arm 3 | 2.0 | 2.0 | 4.00 |
| Arm 4 | 5.0 | 1.5 | 7.50 |
| Arm 5 | 7.0 | 1.5 | 10.50 |

### Optimal Arm
Since the expected mean reward of a Gamma distribution is:

`mean = shape x scale`

the arm with the highest expected reward is:

**Arm 5**, with mean reward **10.50**.

---

## What the Script Does

## 1. Distribution Visualization
For each arm, the script:
- Draws 1000 reward samples from the Gamma distribution.
- Plots a histogram of the sampled rewards.
- Computes the theoretical Gamma PDF.
- Overlays the PDF on the histogram.

This helps compare the shape and spread of the reward distributions.

### Output from this section
- 5 separate plots, one for each arm.
- Each plot shows:
  - sampled reward histogram
  - theoretical Gamma PDF
  - title with arm name, shape, scale, and mean reward

---

## 2. Statistical Analysis
The script then lists the mean reward of each arm and identifies the best one.

### Mean rewards used in the script
- Arm 1: 1.40
- Arm 2: 3.00
- Arm 3: 4.00
- Arm 4: 7.50
- Arm 5: 10.50

### Conclusion
- **Best arm: Arm 5**

---

## 3. Average Reward Analysis
This section compares the learning performance of:
- Epsilon-Greedy with `epsilon = 0.01`
- Epsilon-Greedy with `epsilon = 0.1`
- Epsilon-Greedy with `epsilon = 0.3`
- UCB

### Parameters used
- Maximum steps: `1000`
- Flat-window size for early stopping: `100`
- Tolerance for early stopping: `1e-3`

### How the methods work

#### Epsilon-Greedy
At each time step:
- With probability `epsilon`, the agent explores by choosing a random arm.
- Otherwise, it exploits by choosing the arm with the current highest estimated reward.

#### UCB (Upper Confidence Bound)
At each time step:
- If any arm has not been tried yet, UCB selects it first.
- Otherwise, it selects the arm with the highest value of:

`Q(a) + 2 * sqrt(log(t) / N(a))`

where:
- `Q(a)` = estimated reward of arm `a`
- `N(a)` = number of times arm `a` has been selected
- `t` = current time step

This formula balances:
- exploitation of high-value arms
- exploration of less-tried arms

### What is tracked
- Total reward collected
- Running average reward over time

### Early stopping
The script stops early if the average reward becomes nearly flat in the last 100 time steps.

### Output from this section
- One plot of **Average Reward vs Time**
- Curves for:
  - UCB
  - epsilon = 0.01
  - epsilon = 0.1
  - epsilon = 0.3

### Expected interpretation
Based on the script comments:
- **UCB** reaches high reward fastest and performs best overall.
- **epsilon = 0.1** performs well and learns reasonably fast.
- **epsilon = 0.3** explores too much and performs worse.
- **epsilon = 0.01** learns more slowly in the beginning.

---

## 4. Optimal Action Selection Analysis
This section compares how often each method selects the true optimal arm.

### Process
- The script first computes the true best arm from the known means.
- During training, it records whether the selected action is the optimal one.
- It then computes:

`% optimal actions = 100 * (number of optimal arm selections / total steps)`

### Output from this section
- One plot of **% Optimal Actions vs Time**
- Curves for:
  - UCB
  - epsilon = 0.01
  - epsilon = 0.1
  - epsilon = 0.3

### Expected interpretation
Based on the script comments:
- **UCB** quickly learns the best arm and reaches the highest optimal-action percentage.
- **epsilon = 0.1** also performs well but remains lower because it continues exploring.
- **epsilon = 0.3** stays noticeably lower due to frequent exploration.

---

## Libraries Used
The script imports the following Python libraries:
- `numpy`
- `matplotlib`
- `scipy.special`

---

## Installation
Install the required packages using:

```bash
pip install numpy matplotlib scipy
