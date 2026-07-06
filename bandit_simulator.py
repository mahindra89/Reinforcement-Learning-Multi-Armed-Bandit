from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BanditArm:
    name: str
    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return self.shape * self.scale


DEFAULT_ARMS = [
    BanditArm("Arm 1", 0.7, 2.0),
    BanditArm("Arm 2", 1.5, 2.0),
    BanditArm("Arm 3", 2.0, 2.0),
    BanditArm("Arm 4", 5.0, 1.5),
    BanditArm("Arm 5", 7.0, 1.5),
]


def make_progressive_arms(count: int, spread: float) -> list[BanditArm]:
    """Create a configurable set of Gamma reward arms with increasing means."""
    shapes = np.linspace(0.8, 5.5 + spread, count)
    scales = np.linspace(1.2, 2.0, count)
    return [
        BanditArm(f"Arm {idx + 1}", float(shape), float(scale))
        for idx, (shape, scale) in enumerate(zip(shapes, scales))
    ]


def arm_table(arms: list[BanditArm]) -> list[dict[str, float | str]]:
    return [
        {
            "Arm": arm.name,
            "Shape": round(arm.shape, 3),
            "Scale": round(arm.scale, 3),
            "Expected reward": round(arm.mean, 3),
        }
        for arm in arms
    ]


def simulate_strategy(
    arms: list[BanditArm],
    strategy: str,
    steps: int,
    epsilon: float = 0.1,
    confidence: float = 2.0,
    seed: int | None = 42,
) -> dict[str, np.ndarray | int | float]:
    rng = np.random.default_rng(seed)
    shapes = np.array([arm.shape for arm in arms], dtype=float)
    scales = np.array([arm.scale for arm in arms], dtype=float)
    means = shapes * scales
    best_arm = int(np.argmax(means))
    n_arms = len(arms)

    estimates = np.zeros(n_arms)
    pulls = np.zeros(n_arms)
    rewards = np.zeros(steps)
    selected = np.zeros(steps, dtype=int)
    optimal_hits = np.zeros(steps)

    for step in range(steps):
        if strategy == "UCB":
            untried = np.where(pulls == 0)[0]
            if len(untried):
                action = int(untried[0])
            else:
                bonus = confidence * np.sqrt(np.log(step + 1) / pulls)
                action = int(np.argmax(estimates + bonus))
        elif strategy == "Epsilon-Greedy":
            if rng.random() < epsilon:
                action = int(rng.integers(n_arms))
            else:
                action = int(np.argmax(estimates))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        reward = float(rng.gamma(shapes[action], scales[action]))
        pulls[action] += 1
        estimates[action] += (reward - estimates[action]) / pulls[action]

        selected[step] = action
        rewards[step] = reward
        optimal_hits[step] = 1.0 if action == best_arm else 0.0

    cumulative_reward = np.cumsum(rewards)
    running_average = cumulative_reward / np.arange(1, steps + 1)
    optimal_rate = 100.0 * np.cumsum(optimal_hits) / np.arange(1, steps + 1)

    return {
        "selected": selected,
        "rewards": rewards,
        "running_average": running_average,
        "optimal_rate": optimal_rate,
        "estimates": estimates,
        "pulls": pulls,
        "best_arm": best_arm,
        "total_reward": float(cumulative_reward[-1]),
    }


def compare_strategies(
    arms: list[BanditArm],
    steps: int,
    epsilon: float,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray | int | float]]:
    return {
        "Epsilon-Greedy": simulate_strategy(
            arms=arms,
            strategy="Epsilon-Greedy",
            steps=steps,
            epsilon=epsilon,
            seed=seed,
        ),
        "UCB": simulate_strategy(
            arms=arms,
            strategy="UCB",
            steps=steps,
            confidence=confidence,
            seed=seed,
        ),
    }
