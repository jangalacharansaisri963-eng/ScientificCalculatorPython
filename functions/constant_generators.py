"""
constant_generators.py

Algorithms for generating mathematical constants.
"""

from decimal import Decimal, getcontext


def calculate_pi(digits=None):
    """
    Compute π using a bounded Gauss-Legendre iteration.
    """
    old_prec = getcontext().prec
    if digits is not None:
        getcontext().prec = digits + 5

    try:
        one = Decimal(1)
        two = Decimal(2)
        four = Decimal(4)

        a = one
        b = one / two.sqrt()
        t = Decimal("0.25")
        p = one

        for _ in range(20):
            an = (a + b) / two
            bn = (a * b).sqrt()
            tn = t - p * (a - an) ** 2
            pn = two * p

            if an == a:
                break

            a, b, t, p = an, bn, tn, pn

        result = ((a + b) ** 2) / (four * t)
    finally:
        if digits is not None:
            getcontext().prec = old_prec

    if digits is not None:
        return +result
    return result


def calculate_e(digits=None):
    """
    Compute Euler's number using a bounded series.
    """
    old_prec = getcontext().prec
    if digits is not None:
        getcontext().prec = digits + 5

    try:
        result = Decimal(0)
        factorial = 1
        n = 0

        for _ in range(50):
            if n > 0:
                factorial *= n

            term = Decimal(1) / Decimal(factorial)

            if term == 0:
                break

            result += term
            n += 1
    finally:
        if digits is not None:
            getcontext().prec = old_prec

    if digits is not None:
        return +result
    return result


def calculate_phi(digits=None):
    """
    Golden ratio.
    """
    old_prec = getcontext().prec
    if digits is not None:
        getcontext().prec = digits + 5

    result = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)

    if digits is not None:
        getcontext().prec = old_prec
        return +result
    return result


def calculate_r15(digits=None):
    """
    R15 = Σ 1 / n^(n+1) using a bounded series.
    """
    old_prec = getcontext().prec
    if digits is not None:
        getcontext().prec = digits + 5

    try:
        result = Decimal(0)
        n = 1

        for _ in range(30):
            denominator = Decimal(n) ** Decimal(n + 1)
            term = Decimal(1) / denominator

            if term == 0:
                break

            result += term
            n += 1
    finally:
        if digits is not None:
            getcontext().prec = old_prec

    if digits is not None:
        return +result
    return result

