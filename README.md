# Reinforcement Learning Homework 1 - Multi-Armed Bandit

## Overview
This project implements a simple **multi-armed bandit** problem for Reinforcement Learning homework.  
The code has two main parts:

1. **Designing the bandit**
   - Define 5 bandit arms using Gamma distributions
   - Visualize reward distributions using histograms and theoretical PDFs
   - Compute the mean reward of each arm
   - Identify the optimal arm

2. **Training the agent**
   - Compare **epsilon-greedy** and **UCB (Upper Confidence Bound)** action selection methods
   - Plot **average reward vs time**
   - Plot **percentage of optimal actions vs time**

The code is written in Python and uses NumPy, Matplotlib, and SciPy.

---

## File
- `rla1-2.py` - main Python script for the assignment

---

## Problem Setup

### Bandit Arms
The script defines 5 different arms, each modeled using a **Gamma distribution** with a shape and scale parameter:

- Arm 1: shape = 0.7, scale = 2.0
- Arm 2: shape = 1.5, scale = 2.0
- Arm 3: shape = 2.0, scale = 2.0
- Arm 4: shape = 5.0, scale = 1.5
- Arm 5: shape = 7.0, scale = 1.5

The mean reward of a Gamma distribution is:

\[
\mu = \text{shape} \times \text{scale}
\]

So the arm means are:

- Arm 1: 1.40
- Arm 2: 3.00
- Arm 3: 4.00
- Arm 4: 7.50
- Arm 5: 10.50

**Optimal arm:** Arm 5

---

## What the Code Does

### Task 1.1 - Distribution Visualization
For each arm, the code:
- Samples 1000 rewards from the Gamma distribution
- Plots a histogram of sampled rewards
- Plots the theoretical Gamma PDF on top of the histogram

This helps visualize how reward distributions differ from one arm to another.

### Task 1.2 - Statistical Analysis
The code calculates the expected mean reward of each arm and identifies the best arm based on the highest mean.

### Task 2.1 - Average Reward Analysis
The code compares:
- Epsilon-greedy with:
  - epsilon = 0.01
  - epsilon = 0.1
  - epsilon = 0.3
- UCB

For each method, it:
- Chooses an arm at each step
- Samples a reward
- Updates the estimated reward of the selected arm
- Tracks the running average reward over time
- Stops early if the last 100 values are nearly flat

Then it plots:
- **Average reward vs game time**

### Task 2.2 - Optimal Action Selection
The code also tracks how often the algorithm chooses the true optimal arm.

Then it plots:
- **% optimal actions vs game time**

---

## Algorithms Used

### 1. Epsilon-Greedy
Epsilon-greedy balances exploration and exploitation:
- With probability **epsilon**, choose a random arm
- Otherwise, choose the arm with the highest estimated reward

This project tests three epsilon values:
- 0.01
- 0.1
- 0.3

### 2. Upper Confidence Bound (UCB)
UCB selects actions using both:
- Estimated reward
- Uncertainty of the estimate

It first tries each arm once, then uses:

\[
UCB(a) = Q(a) + 2 \sqrt{\frac{\log(t)}{N(a)}}
\]

where:
- \(Q(a)\) is the estimated reward of arm \(a\)
- \(N(a)\) is the number of times arm \(a\) has been selected
- \(t\) is the current time step

This usually helps the agent find the best arm faster.

---

## Requirements

Install the following Python packages before running the script:

```bash
pip install numpy matplotlib scipy
```

---

## How to Run

Run the script using Python:

```bash
python rla1-2.py
```

If you are using Jupyter Notebook or Google Colab, you can also run the cells step by step.

---

## Output
The script generates:

1. **Five reward distribution plots**
   - One for each bandit arm
   - Histogram + theoretical PDF

2. **Average reward plot**
   - Comparison of UCB and epsilon-greedy strategies

3. **Optimal action percentage plot**
   - Comparison of how often each method selects the best arm

It also prints:
- The best arm based on the true mean reward

---

## Observations
Based on the code comments and results:

- **UCB** learns the best arm quickly and reaches high reward faster
- **Epsilon-greedy with epsilon = 0.1** performs reasonably well
- **Epsilon = 0.3** explores too much and gives lower performance
- **Epsilon = 0.01** improves more slowly because it explores less in the beginning

For optimal action selection:
- **UCB** reaches the highest percentage of optimal actions
- **Epsilon = 0.1** is good but slightly lower
- **Epsilon = 0.3** stays lower due to too much exploration

---

## Notes
- The random seed is set using `np.random.seed(0)` for reproducibility in Task 2.
- The script uses an **early stopping condition** if the last 100 values are nearly flat.
- The code is suitable for understanding the difference between exploration strategies in bandit problems.

---

## Author
Prepared for:

**CS 4391/5391 - Reinforcement Learning**  
**Homework 1**
