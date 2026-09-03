"""Shared statistical helpers for judging whether a result is distinguishable
from chance, not just a point estimate that could easily be noise at small
sample sizes -- exactly the failure mode the Accuracy page and backtest
IC exist to avoid overclaiming on, especially early when n is still small.
"""
from __future__ import annotations

from scipy import stats as scipy_stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Confidence interval for a binomial proportion (e.g. prediction hit
    rate). Wilson's interval, not the naive normal-approximation one --
    the naive version can report nonsensical bounds (below 0% or above
    100%) at exactly the small sample sizes this is most needed for.
    Returns None when there's no data to bound.
    """
    if n == 0:
        return None
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def ic_significance(ic: float, n: int) -> float | None:
    """Two-tailed p-value for the null hypothesis that the true information
    coefficient is 0 -- i.e. the score has no real ranking skill and the
    measured IC is just what n points of pure noise would produce some of
    the time. Standard t-test for a Pearson correlation coefficient. None
    when n is too small (or ic is degenerate) for the test to apply.
    """
    if n < 3 or abs(ic) >= 1:
        return None
    df = n - 2
    t_stat = ic * (df**0.5) / ((1 - ic**2) ** 0.5)
    return float(2 * scipy_stats.t.sf(abs(t_stat), df))
