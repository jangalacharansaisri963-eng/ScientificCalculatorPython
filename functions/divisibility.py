"""Divisibility helper functions for ScientificCalculator

This module provides:
- nearest_divisible(n, d): returns the nearest integer to n that is divisible by d (tie -> smaller)
- is_divisible(n, d): simple boolean check
- divisibility_tests(n, divisors=None): runs tests for a set of divisors and returns a mapping

Default divisors include: 2,3,4,5,6,7,8,9,10,11,25,125,13
Plus 10 additional common divisors: 12,14,15,16,17,18,19,20,21,24

Usage example:
>>> from functions.divisibility import divisibility_tests
>>> divisibility_tests(100)
{2: {'divisible': True, 'nearest': 100}, ...}
"""
from typing import List, Dict, Tuple

DEFAULT_DIVISORS: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25, 125, 13]
# Ten extra divisors chosen to cover common cases
EXTRA_DIVISORS: List[int] = [12, 14, 15, 16, 17, 18, 19, 20, 21, 24]
ALL_DIVISORS: List[int] = DEFAULT_DIVISORS + EXTRA_DIVISORS


def is_divisible(n: int, d: int) -> bool:
    """Return True if n is divisible by d.

    Raises ValueError if d is 0.
    """
    if d == 0:
        raise ValueError("Divisor d must not be 0")
    return n % d == 0


def nearest_divisible(n: int, d: int) -> int:
    """Return the nearest integer to n that is divisible by d.

    If n is already divisible by d, returns n. If there's a tie between two
    candidates (e.g., n=7, d=4 -> candidates 4 and 8 are both distance 3 and 1
    respectively, no tie), this function prefers the smaller candidate.

    Examples:
    >>> nearest_divisible(10, 3)
    9
    >>> nearest_divisible(11, 4)
    12
    >>> nearest_divisible(12, 4)
    12
    """
    if d == 0:
        raise ValueError("Divisor d must not be 0")

    # If already divisible
    if n % d == 0:
        return n

    # Compute lower and higher candidates
    lower = n - (n % d)
    higher = lower + d

    # Distances
    dist_lower = abs(n - lower)
    dist_higher = abs(higher - n)

    # Prefer the nearest; on tie prefer the smaller (lower)
    if dist_lower <= dist_higher:
        return lower
    return higher


def divisibility_tests(n: int, divisors: List[int] = None) -> Dict[int, Dict[str, object]]:
    """Run divisibility tests for n against a list of divisors.

    Returns a mapping from divisor -> { 'divisible': bool, 'nearest': int }
    """
    if divisors is None:
        divisors = ALL_DIVISORS
    results: Dict[int, Dict[str, object]] = {}
    for d in divisors:
        try:
            divisible = is_divisible(n, d)
            nearest = n if divisible else nearest_divisible(n, d)
        except ValueError:
            # skip invalid divisor like 0 but preserve the entry
            results[d] = {"divisible": False, "nearest": None}
            continue
        results[d] = {"divisible": divisible, "nearest": nearest}
    return results


def format_divisibility_report(n: int, divisors: List[int] = None) -> str:
    """Return a human-readable multi-line report for divisibility tests.

    Example output:
    100 is divisible by 2
    100 is divisible by 3 -> nearest divisible is 99
    """
    results = divisibility_tests(n, divisors)
    lines: List[str] = []
    for d in sorted(results.keys()):
        r = results[d]
        if r["divisible"]:
            lines.append(f"{n} is divisible by {d}")
        else:
            nearest = r["nearest"]
            lines.append(f"{n} is NOT divisible by {d}; nearest divisible: {nearest}")
    return "\n".join(lines)


