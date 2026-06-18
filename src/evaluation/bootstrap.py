from typing import Callable, Dict, List

import numpy as np
from tqdm import tqdm


def bootstrap_sample(items: List, rng: np.random.Generator) -> List:
    if len(items) == 0:
        raise ValueError("Cannot bootstrap an empty list.")

    indices = rng.integers(0, len(items), size=len(items))
    return [items[i] for i in indices]


def summarize_bootstrap(samples: List[float]) -> Dict[str, float]:
    if len(samples) == 0:
        raise ValueError("Cannot summarize empty bootstrap samples.")

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
    }


def bootstrap_metric(
    metric_fn: Callable[[List], float],
    trajectories: List,
    n_samples: int,
    seed: int,
    desc: str = "Bootstrap",
) -> Dict[str, float]:
    if n_samples <= 1:
        raise ValueError(f"n_samples must be greater than 1, got {n_samples}")

    rng = np.random.default_rng(seed)
    values = []

    for _ in tqdm(range(n_samples), desc=desc, leave=False):
        sample = bootstrap_sample(trajectories, rng)
        values.append(float(metric_fn(sample)))

    return summarize_bootstrap(values)


def bootstrap_two_group_metric(
    metric_fn: Callable[[List, List], float],
    first: List,
    second: List,
    n_samples: int,
    seed: int,
    desc: str = "Bootstrap",
) -> Dict[str, float]:
    if n_samples <= 1:
        raise ValueError(f"n_samples must be greater than 1, got {n_samples}")

    rng = np.random.default_rng(seed)
    values = []

    for _ in tqdm(range(n_samples), desc=desc, leave=False):
        first_sample = bootstrap_sample(first, rng)
        second_sample = bootstrap_sample(second, rng)
        values.append(float(metric_fn(first_sample, second_sample)))

    return summarize_bootstrap(values)
