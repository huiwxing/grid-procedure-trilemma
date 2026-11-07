"""Cluster bootstrap.

`boot(vectors, cluster_of, key)`: arms = item -> value vectors (correctness averaged over repeats, or any per-cell
number); items = those every arm has (sorted ids), so arms are always compared on the same items. A replicate draws the
clusters with replacement (as many as there are) and takes each arm's mean over the items of the drawn clusters, a
cluster counted as often as it is drawn. Point = the arm means over all common items; `reps` = REPLICATES replicate
means per arm (None when fewer than 2 clusters: no interval).

Every estimate's generator is seeded from its own key (sha256 of a string naming the estimate), so an estimate does not
move when other estimates are added or reordered, and a rerun gives the same numbers.

`model_weights`: the model level of the crossed bootstrap (models x clusters: the reader models of a tier are a
random effect, so a tier interval covers the choice of models as well as the choice of items): per replicate the
models of a tier are drawn with replacement, on top of the shared cluster draw; a tier mean is the drawn models' mean.

`interval`: point estimate with the 2.5 / 97.5 percentiles of the replicates (percentile interval, 95%).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

REPLICATES = 2000
LEVEL = 0.95
DIGITS = 4


def seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def r4(x) -> float | None:
    return None if x is None else round(float(x), DIGITS)


@dataclass(frozen=True)
class Boot:
    items: tuple                 # the common item ids, sorted
    n_clusters: int
    point: np.ndarray            # (arms,) means over the common items
    reps: np.ndarray | None      # (REPLICATES, arms) replicate means; None below 2 clusters
    rng: np.random.Generator     # continues the estimate's stream (model draws)


def boot(vectors: list[dict], cluster_of: dict[str, str], key: str, replicates: int = REPLICATES) -> Boot:
    rng = np.random.default_rng(seed_of(key))
    common = sorted(set.intersection(*(set(v) for v in vectors))) if vectors else []
    if not common:
        return Boot((), 0, np.full(len(vectors), np.nan), None, rng)
    names = sorted({cluster_of[i] for i in common})
    index = {c: k for k, c in enumerate(names)}
    idx = np.array([index[cluster_of[i]] for i in common])
    values = np.array([[v[i] for i in common] for v in vectors], dtype=float)          # (arms, items)
    sums = np.stack([np.bincount(idx, weights=row, minlength=len(names)) for row in values])   # (arms, clusters)
    sizes = np.bincount(idx, minlength=len(names)).astype(float)
    point = values.mean(axis=1)
    if len(names) < 2:
        return Boot(tuple(common), len(names), point, None, rng)
    draws = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=replicates)   # (B, clusters)
    reps = (draws @ sums.T) / (draws @ sizes)[:, None]
    return Boot(tuple(common), len(names), point, reps, rng)


def model_weights(rng: np.random.Generator, n_models: int, replicates: int = REPLICATES) -> np.ndarray:
    """(replicates, n_models) weights summing to 1 per replicate: models drawn with replacement."""
    return rng.multinomial(n_models, np.full(n_models, 1.0 / n_models), size=replicates) / n_models


def interval(point, reps: np.ndarray | None) -> dict:
    if point is None or not np.isfinite(point):
        return {"est": None, "ci": None}
    if reps is None:
        return {"est": r4(point), "ci": None}
    lo, hi = np.quantile(reps, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return {"est": r4(point), "ci": [r4(lo), r4(hi)]}
