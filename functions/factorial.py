"""
factorial.py

Factorial functions implemented using integers for calculation and
converted to Decimal via Decimal(str(...)) only for input parsing and output.

This version intentionally avoids using Decimal arithmetic operations
(like power on Decimal, to_integral_value, etc.). It only uses
Decimal(str(...)) to parse inputs reliably and getcontext can be set
externally for overall Decimal behavior.
"""

from decimal import Decimal, getcontext

# Users may set precision externally: getcontext().prec = 50


def _to_decimal(value):
    """
    Convert value to Decimal using Decimal(str(value)) to avoid float
    representation issues. This is the only Decimal "amenity" used.
    """
    return Decimal(str(value))


def _is_integer_decimal(dec):
    """
    Check if a Decimal created from a value corresponds exactly to an integer
    without using Decimal's to_integral_value or other conveniences.
    """
    try:
        i = int(dec)
    except (ValueError, TypeError, OverflowError):
        return False

    return Decimal(str(i)) == dec


def _factorial_int(n):
    """Compute n! as a plain Python integer."""
    if n < 0:
        raise ValueError("Factorial is only defined for non-negative integers.")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def factorial(x):
    """
    Standard factorial: n! for integer n >= 0

    Inputs are parsed with Decimal(str(x)) but the heavy lifting is done
    with integer arithmetic. The result is returned as Decimal(str(result)).
    """
    x_dec = _to_decimal(x)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Factorial is only defined for integers.")

    n = int(x_dec)

    result_int = _factorial_int(n)

    return Decimal(str(result_int))


def doublefactorial(x):
    """
    n!!  (double factorial)

    Example:
    8!! = 8×6×4×2
    """
    x_dec = _to_decimal(x)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Double factorial is only defined for integers.")

    n = int(x_dec)

    if n < 0:
        raise ValueError("Negative numbers not allowed.")

    if n == 0 or n == 1:
        return Decimal('1')

    result_int = 1
    while n > 1:
        result_int *= n
        n -= 2

    return Decimal(str(result_int))


def superfactorial(x):
    """
    sf(n) = 1! × 2! × ... × n!
    """
    x_dec = _to_decimal(x)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Superfactorial index must be an integer.")

    n = int(x_dec)

    result_int = 1
    for i in range(1, n + 1):
        result_int *= _factorial_int(i)

    return Decimal(str(result_int))


def hyperfactorial(x):
    """
    H(n) = 1^1 × 2^2 × ... × n^n

    Computed using integer pow for exactness, then converted to Decimal.
    """
    x_dec = _to_decimal(x)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Hyperfactorial index must be an integer.")

    n = int(x_dec)

    result_int = 1
    for i in range(1, n + 1):
        result_int *= pow(i, i)

    return Decimal(str(result_int))


def primefactorial(x):
    """
    Product of all primes <= n
    """
    x_dec = _to_decimal(x)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Prime factorial index must be an integer.")

    n = int(x_dec)

    result_int = 1

    for i in range(2, n + 1):
        prime = True
        limit = int(i ** 0.5) + 1
        for j in range(2, limit):
            if i % j == 0:
                prime = False
                break
        if prime:
            result_int *= i

    return Decimal(str(result_int))


def risingfactorial(x, k):
    """
    (x)^k = x (x+1) ... (x+k-1)

    Inputs are parsed with Decimal(str(...)), but both x and k must be
    integer-valued in this implementation. Multiplication is done using
    integer arithmetic and converted at the end.
    """
    x_dec = _to_decimal(x)
    k_dec = _to_decimal(k)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Rising factorial x must be an integer in this implementation.")
    if not _is_integer_decimal(k_dec):
        raise ValueError("Rising factorial k must be an integer.")

    x_int = int(x_dec)
    k_int = int(k_dec)

    result_int = 1
    for i in range(k_int):
        result_int *= (x_int + i)

    return Decimal(str(result_int))


def fallingfactorial(x, k):
    """
    x_(k) = x (x-1) ... (x-k+1)

    Inputs are parsed with Decimal(str(...)), but both x and k must be
    integer-valued in this implementation. Multiplication is done using
    integer arithmetic and converted at the end.
    """
    x_dec = _to_decimal(x)
    k_dec = _to_decimal(k)

    if not _is_integer_decimal(x_dec):
        raise ValueError("Falling factorial x must be an integer in this implementation.")
    if not _is_integer_decimal(k_dec):
        raise ValueError("Falling factorial k must be an integer.")

    x_int = int(x_dec)
    k_int = int(k_dec)

    result_int = 1
    for i in range(k_int):
        result_int *= (x_int - i)

    return Decimal(str(result_int))
