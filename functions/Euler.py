"""
Euler.py

Leonhard Euler (1707–1783) and related mathematical objects.

This module provides:

  • Euler's number e  (base of the natural logarithm)
  • Euler's identity   e^(iπ) + 1 = 0
  • Euler's formula    e^(ix) = cos(x) + i·sin(x)
  • Euler-Mascheroni constant γ
  • Euler's totient function φ(n)
  • Euler numbers E_n (secant / zigzag numbers)
  • Euler polynomials
  • Classic series and identities discovered by Euler
    (Basel problem, product formulae, continued fractions, …)
  • A few results obtained with collaborators (Goldbach, Bernoulli, …)

All implementations are pure Python (stdlib + optional high-precision Decimal).
No external symbolic engine is required.
"""

from __future__ import annotations

import math
import cmath
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Union, List, Tuple, Dict, Any

Number = Union[int, float, complex, Decimal]

# ---------------------------------------------------------------------------
# High-precision helpers
# ---------------------------------------------------------------------------

def _set_prec(digits: int = 50) -> None:
    getcontext().prec = max(digits + 5, 28)


def e_series(digits: int = 30) -> Decimal:
    """
    Compute Euler's number e = Σ 1/n!  to the requested number of decimal digits.
    """
    _set_prec(digits)
    e = Decimal(1)
    term = Decimal(1)
    n = 1
    # Continue until the term is smaller than the desired precision
    while True:
        term /= Decimal(n)
        if term == 0:
            break
        e += term
        n += 1
        if n > digits * 3:          # safety
            break
    return +e                      # unary + applies current precision


def euler_mascheroni(digits: int = 30) -> Decimal:
    """
    Approximate the Euler-Mascheroni constant

        γ = lim (H_n - ln n)

    using a simple accelerated series (works well for modest precision).
    For higher precision a more sophisticated algorithm would be used.
    """
    _set_prec(digits)
    # Simple definition with a reasonably large n
    n = 10 ** min(6, digits // 2 + 3)
    H = Decimal(0)
    for k in range(1, n + 1):
        H += Decimal(1) / Decimal(k)
    gamma_constant = H - Decimal(n).ln()
    return +gamma_constant


# ---------------------------------------------------------------------------
# Core constants (float for everyday use, Decimal for high precision)
# ---------------------------------------------------------------------------

E = math.e
EULER = E                                 # alias
EULER_NUMBER = E

# Euler-Mascheroni constant (also already present in constants.py)
GAMMA_CONSTANT = 0.57721566490153286060651209008240243104215933593992
EULER_MASCHERONI = GAMMA_CONSTANT
EULER_CONSTANT = GAMMA_CONSTANT

# ---------------------------------------------------------------------------
# Euler's identity and formula
# ---------------------------------------------------------------------------

def euler_identity() -> complex:
    """
    Evaluate Euler's identity:

        e^(iπ) + 1  =  0

    Returns the complex value (should be extremely close to 0).
    """
    return cmath.exp(1j * math.pi) + 1


def euler_formula(x: Number) -> complex:
    """
    Euler's formula:

        e^(i x)  =  cos(x) + i·sin(x)

    Accepts real or complex x; returns the complex exponential.
    """
    if isinstance(x, Decimal):
        x = float(x)
    return cmath.exp(1j * x)


def euler_formula_components(x: float) -> Tuple[float, float, complex]:
    """
    Return (cos(x), sin(x), e^(ix)) so the identity can be inspected numerically.
    """
    c = math.cos(x)
    s = math.sin(x)
    return c, s, complex(c, s)


def verify_euler_identity(tol: float = 1e-12) -> bool:
    """Return True if |e^(iπ) + 1| < tol."""
    return abs(euler_identity()) < tol


# ---------------------------------------------------------------------------
# Euler's totient function φ(n)
# ---------------------------------------------------------------------------

def euler_totient(n: int) -> int:
    """
    Euler's totient function φ(n) = number of integers k in 1..n
    that are coprime to n.

    Classic product formula:
        φ(n) = n · Π (1 - 1/p)   over distinct prime factors p of n.
    """
    if n < 0:
        raise ValueError("totient is defined for non-negative integers")
    if n <= 1:
        return n
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def euler_totient_sum(n: int) -> int:
    """Σ_{k=1..n} φ(k).  Useful for verifying Σ_{d|n} φ(d) = n."""
    return sum(euler_totient(k) for k in range(1, n + 1))


# ---------------------------------------------------------------------------
# Euler numbers E_n  (secant / zigzag numbers)
# ---------------------------------------------------------------------------

def euler_number(n: int) -> int:
    """
    The Euler numbers E_n (also called secant numbers when n even).

    They appear in the Taylor series:

        sech(x) = Σ E_{2k} · x^{2k} / (2k)!

        sec(x)  = Σ |E_{2k}| · x^{2k} / (2k)!

    Computed via the classic recurrence.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n % 2 == 1:
        return 0                       # odd Euler numbers vanish
    # Use a recurrence based on the generating function
    # E_0 = 1
    # Σ_{k=0}^{m} C(2m, 2k) E_{2k} = 0  for m > 0
    E = [0] * (n + 1)
    E[0] = 1
    for m in range(1, n // 2 + 1):
        s = 0
        for k in range(m):
            s += math.comb(2 * m, 2 * k) * E[2 * k]
        E[2 * m] = -s
    return E[n]


def euler_numbers_up_to(n: int) -> List[int]:
    """Return the list [E_0, E_1, …, E_n]."""
    return [euler_number(k) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# Euler polynomials (basic evaluation)
# ---------------------------------------------------------------------------

def euler_polynomial(n: int, x: float) -> float:
    """
    Evaluate the n-th Euler polynomial E_n(x).

    Generating function:  2 e^{xt} / (e^t + 1) = Σ E_n(x) t^n / n!
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    # Simple explicit formula using finite differences / Bernoulli relation
    # E_n(x) = Σ_{k=0}^n C(n,k) · (E_k / 2^k) · (x - 1/2)^{n-k}   (approx form)
    # For exact small-n we use the recurrence:
    # E_0(x) = 1
    # E_n'(x) = n E_{n-1}(x)
    # ∫_0^1 E_n(x) dx = 0  for n ≥ 1
    # A practical implementation for modest n:
    from math import comb
    # Use the relation with Bernoulli polynomials would be ideal,
    # but a direct power-sum style works for demonstration.
    # Here we use the explicit sum:
    # E_n(x) = Σ_{k=0}^n (1/2^k) * C(n,k) * E_k * (x-1/2)^{n-k}
    # where E_k are the Euler numbers above.
    s = 0.0
    for k in range(n + 1):
        Ek = euler_number(k)
        s += (Ek / (2 ** k)) * comb(n, k) * ((x - 0.5) ** (n - k))
    return s


# ---------------------------------------------------------------------------
# Classic Euler identities & series
# ---------------------------------------------------------------------------

def basel_sum(terms: int = 10000) -> float:
    """
    Euler's solution of the Basel problem:

        Σ_{n=1}^∞ 1/n²  =  π² / 6

    Returns the partial sum (demonstrates convergence toward π²/6).
    """
    return sum(1.0 / (k * k) for k in range(1, terms + 1))


def basel_exact() -> float:
    """Return the exact closed form π²/6."""
    return math.pi ** 2 / 6


def euler_product_zeta2(primes: List[int] | None = None) -> float:
    """
    Euler product for ζ(2):

        π²/6 = Π_p  1 / (1 - p^{-2})
    """
    if primes is None:
        # first few primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    prod = 1.0
    for p in primes:
        prod *= 1.0 / (1.0 - 1.0 / (p * p))
    return prod


def euler_continued_fraction_e(terms: int = 20) -> Fraction:
    """
    Euler's continued fraction for e:

        e = 2 + 1/(1 + 1/(2 + 1/(1 + 1/(1 + 1/(4 + 1/(1 + 1/(1 + 1/(6 + …))))))))

    The pattern of partial quotients is [2; 1,2,1, 1,4,1, 1,6,1, …].
    Returns a Fraction approximation.
    """
    # Build the list of partial quotients
    a = [2]
    k = 1
    while len(a) < terms:
        a.extend([1, 2 * k, 1])
        k += 1
    a = a[:terms]

    # Evaluate continued fraction from the tail
    frac = Fraction(a[-1])
    for coeff in reversed(a[:-1]):
        frac = coeff + 1 / frac
    return frac


def exp_series(x: Number, terms: int = 30) -> complex:
    """
    Taylor series for e^x = Σ x^n / n!   (works for complex x).
    """
    if isinstance(x, Decimal):
        x = complex(float(x))
    elif not isinstance(x, complex):
        x = complex(x)
    s = 0j
    term = 1j * 0 + 1
    for n in range(terms):
        s += term
        term *= x / (n + 1)
    return s


# ---------------------------------------------------------------------------
# Collaboration / historical notes (as callable documentation)
# ---------------------------------------------------------------------------

def euler_goldbach_conjecture_note() -> str:
    """
    Euler corresponded extensively with Christian Goldbach.
    The Goldbach conjecture (every even integer > 2 is the sum of two primes)
    appears in their letters; Euler verified it for many cases.
    """
    return (
        "Euler–Goldbach correspondence: Euler examined Goldbach's claim that "
        "every even integer greater than 2 can be written as the sum of two primes. "
        "While a full proof is still open, Euler supplied extensive numerical evidence."
    )


def euler_bernoulli_note() -> str:
    """
    Euler built heavily on the work of the Bernoulli family,
    especially on the Bernoulli numbers that appear in the Euler–Maclaurin formula.
    """
    return (
        "Euler–Maclaurin formula connects sums and integrals and involves Bernoulli numbers. "
        "Euler also gave the famous evaluation ζ(2k) in terms of Bernoulli numbers."
    )


def euler_identity_statement() -> str:
    """Return the classic statement of Euler's identity."""
    return "e^(iπ) + 1 = 0"


def euler_formula_statement() -> str:
    """Return the classic statement of Euler's formula."""
    return "e^(i x) = cos(x) + i·sin(x)"


# ---------------------------------------------------------------------------
# Convenience registry for the calculator
# ---------------------------------------------------------------------------

EULER_FUNCTIONS = {
    # constants
    "e": E,
    "euler": E,
    "E": E,
    "gamma_constant": GAMMA_CONSTANT,
    "euler_gamma": GAMMA_CONSTANT,
    "euler_mascheroni": GAMMA_CONSTANT,

    # functions
    "euler_identity": euler_identity,
    "euler_formula": euler_formula,
    "euler_totient": euler_totient,
    "phi": euler_totient,                 # common alias
    "euler_number": euler_number,
    "euler_poly": euler_polynomial,
    "basel": basel_sum,
    "basel_exact": basel_exact,
    "e_series": e_series,
    "exp_series": exp_series,
    "verify_euler_identity": verify_euler_identity,
}

__all__ = [
    "E", "EULER", "EULER_NUMBER",
    "GAMMA_CONSTANT", "EULER_MASCHERONI", "EULER_CONSTANT",
    "e_series", "euler_mascheroni",
    "euler_identity", "euler_formula", "euler_formula_components",
    "verify_euler_identity",
    "euler_totient", "euler_totient_sum",
    "euler_number", "euler_numbers_up_to",
    "euler_polynomial",
    "basel_sum", "basel_exact", "euler_product_zeta2",
    "euler_continued_fraction_e", "exp_series",
    "euler_goldbach_conjecture_note", "euler_bernoulli_note",
    "euler_identity_statement", "euler_formula_statement",
    "EULER_FUNCTIONS",
]
