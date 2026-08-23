"""
logarithms.py

Decimal-only logarithmic functions and identities for the scientific calculator.
No use of the math module — all calculations use decimal.getcontext() operations
for higher precision and reproducibility.

This module provides:
- ln, log (change-of-base), log10, log2, log1p, decimal_pow
- helper utilities: to_decimal, set_precision
- 20 logarithmic identity functions (renamed without the 'identity_' prefix)
  which compute both sides of each identity and return (lhs, rhs) as Decimals
  for inspection or comparison.

Notes:
- All inputs are converted to Decimal without going through float.
- Requires a decimal.Context with methods ln, log10, exp (available in modern
  Python's decimal implementation). If those methods are not available, the
  module will raise a RuntimeError recommending a Python upgrade.
"""

from decimal import Decimal, getcontext, InvalidOperation
from typing import Tuple, List

_ctx = getcontext()

# Quick checks for required context methods. If missing, fail early.
if not (hasattr(_ctx, "ln") and hasattr(_ctx, "log10") and hasattr(_ctx, "exp")):
    raise RuntimeError(
        "The current decimal context does not provide ln/log10/exp. "
        "Please use a Python version whose decimal.Context implements these functions."
    )


def set_precision(prec: int) -> None:
    """Set decimal precision for subsequent calculations.

    Example: set_precision(50)
    """
    if not isinstance(prec, int) or prec <= 0:
        raise ValueError("prec must be a positive integer")
    _ctx.prec = prec


def to_decimal(x) -> Decimal:
    """Convert input to Decimal without intermediate float conversions.

    Accepts int, str, Decimal, or anything convertible to Decimal via the
    Decimal constructor.
    """
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(x)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TypeError(f"Cannot convert {x!r} to Decimal: {exc}")


# Core logarithmic functions using decimal.Context methods
def ln(x) -> Decimal:
    """Natural logarithm (base e) using the decimal context.

    Domain: x > 0.
    """
    xd = to_decimal(x)
    if xd <= 0:
        raise ValueError(f"ln domain error: x must be > 0 (got {x})")
    return _ctx.ln(xd)


def log(base, value) -> Decimal:
    """Logarithm of `value` with given `base`.

    Uses change-of-base: log_base(value) = ln(value) / ln(base).
    Domain: base > 0, base != 1, value > 0.
    """
    bd = to_decimal(base)
    vd = to_decimal(value)
    if bd <= 0 or bd == 1:
        raise ValueError(f"base must be > 0 and != 1 (got {base})")
    if vd <= 0:
        raise ValueError(f"value must be > 0 (got {value})")
    # Use context ln for both for best precision
    return _ctx.ln(vd) / _ctx.ln(bd)


def log10(value) -> Decimal:
    """Common logarithm base 10."""
    vd = to_decimal(value)
    if vd <= 0:
        raise ValueError(f"log10 domain error: value must be > 0 (got {value})")
    return _ctx.log10(vd)


def log2(value) -> Decimal:
    """Logarithm base 2."""
    return log(Decimal(2), value)


def log1p(x) -> Decimal:
    """Accurate computation of ln(1 + x). Domain: x > -1."""
    xd = to_decimal(x)
    if xd <= -1:
        raise ValueError(f"log1p domain error: x must be > -1 (got {x})")
    return _ctx.ln(Decimal(1) + xd)


def decimal_pow(x, y) -> Decimal:
    """Compute x ** y for Decimal x and y using exp(y * ln(x)).

    Handles positive x. For x == 0 and integer y >= 0, returns 0.
    """
    xd = to_decimal(x)
    yd = to_decimal(y)
    if xd == 0:
        # 0 ** y: defined for non-negative integer y; otherwise Error
        if yd == 0:
            # 0**0 is treated as 1 in many contexts; keep it undefined here
            raise ValueError("0 ** 0 is undefined")
        # If y is an integer >= 1, return 0
        try:
            if yd == int(yd) and yd >= 1:
                return Decimal(0)
        except (InvalidOperation, TypeError):
            pass
        raise ValueError("0 ** y undefined for non-integer or negative y")
    if xd < 0:
        # Negative base with non-integer exponent would be complex
        try:
            if yd == int(yd):
                # integer exponent OK
                return xd.__pow__(int(yd))
        except (InvalidOperation, TypeError):
            pass
        raise ValueError("Negative base with non-integer exponent would be complex")
    # General positive base: exp(y * ln(x))
    return _ctx.exp(yd * _ctx.ln(xd))


# Renamed identity functions (removed 'identity_' prefix). Each returns (lhs, rhs).


def log_mul(base, x, y) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x * y) = log_b(x) + log_b(y)"""
    lhs = log(base, to_decimal(x) * to_decimal(y))
    rhs = log(base, x) + log(base, y)
    return lhs, rhs


def log_div(base, x, y) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x / y) = log_b(x) - log_b(y)"""
    lhs = log(base, to_decimal(x) / to_decimal(y))
    rhs = log(base, x) - log(base, y)
    return lhs, rhs


def log_pow(base, x, r) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x ** r) = r * log_b(x)"""
    lhs = log(base, decimal_pow(x, r))
    rhs = to_decimal(r) * log(base, x)
    return lhs, rhs


def change_of_base(new_base, x) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x) = ln(x) / ln(b)"""
    lhs = log(new_base, x)
    rhs = _ctx.ln(to_decimal(x)) / _ctx.ln(to_decimal(new_base))
    return lhs, rhs


def log_reciprocal(base, x) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(1 / x) = -log_b(x)"""
    lhs = log(base, Decimal(1) / to_decimal(x))
    rhs = -log(base, x)
    return lhs, rhs


def log_of_base(base) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(b) = 1"""
    lhs = log(base, base)
    rhs = Decimal(1)
    return lhs, rhs


def log_of_one(base) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(1) = 0"""
    lhs = log(base, Decimal(1))
    rhs = Decimal(0)
    return lhs, rhs


def log_sqrt(base, x) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(sqrt(x)) = (1/2) * log_b(x)"""
    lhs = log(base, decimal_pow(x, Decimal(1) / Decimal(2)))
    rhs = (Decimal(1) / Decimal(2)) * log(base, x)
    return lhs, rhs


def log_base_power(base, k, x) -> Tuple[Decimal, Decimal]:
    """Identity: log_{b**k}(x) = log_b(x) / k"""
    new_base = decimal_pow(base, k)
    lhs = log(new_base, x)
    rhs = log(base, x) / to_decimal(k)
    return lhs, rhs


def exp_log_inversion(base, x) -> Tuple[Decimal, Decimal]:
    """Identity: b ** log_b(x) = x (for x > 0)"""
    lhs = decimal_pow(base, log(base, x))
    rhs = to_decimal(x)
    return lhs, rhs


def log_chain(a, b, c) -> Tuple[Decimal, Decimal]:
    """Identity: log_a(b) * log_b(c) = log_a(c)"""
    lhs = log(a, b) * log(b, c)
    rhs = log(a, c)
    return lhs, rhs


def reciprocal_change(base, x) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x) = 1 / log_x(b)"""
    lhs = log(base, x)
    rhs = Decimal(1) / log(x, base)
    return lhs, rhs


def log_power_sum(base, x, y, p, q) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x^p * y^q) = p log_b(x) + q log_b(y)"""
    lhs = log(base, decimal_pow(x, p) * decimal_pow(y, q))
    rhs = to_decimal(p) * log(base, x) + to_decimal(q) * log(base, y)
    return lhs, rhs


def log_root(base, x, n) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(root_n(x)) = log_b(x) / n"""
    # root_n(x) = x ** (1/n)
    lhs = log(base, decimal_pow(x, Decimal(1) / to_decimal(n)))
    rhs = log(base, x) / to_decimal(n)
    return lhs, rhs


def product_of_powers(base, terms: List[Tuple]) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(prod_i x_i^{p_i}) = sum_i p_i * log_b(x_i).

    `terms` should be an iterable of (x_i, p_i).
    Returns (lhs, rhs).
    """
    prod = Decimal(1)
    rhs_sum = Decimal(0)
    for x_i, p_i in terms:
        prod *= decimal_pow(x_i, p_i)
        rhs_sum += to_decimal(p_i) * log(base, x_i)
    lhs = log(base, prod)
    return lhs, rhs_sum


def log_negative_power(base, x, r) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x ** -r) = -r * log_b(x)"""
    lhs = log(base, decimal_pow(x, -to_decimal(r)))
    rhs = -to_decimal(r) * log(base, x)
    return lhs, rhs


def sum_to_log(base, x, y) -> Tuple[Decimal, Decimal]:
    """Convert product inside log to sum: log_b(xy) = log_b(x) + log_b(y) (alias)"""
    return log_mul(base, x, y)


def change_base_via_10(base, x) -> Tuple[Decimal, Decimal]:
    """Use base 10: log_b(x) = log10(x) / log10(b)"""
    lhs = log(base, x)
    rhs = log10(x) / log10(base)
    return lhs, rhs


def log_of_power_product(base, x, y, r) -> Tuple[Decimal, Decimal]:
    """Identity: log_b(x^r * y^r) = r (log_b(x) + log_b(y))"""
    lhs = log(base, decimal_pow(x, r) * decimal_pow(y, r))
    rhs = to_decimal(r) * (log(base, x) + log(base, y))
    return lhs, rhs


def log_scaling(base, x, k) -> Tuple[Decimal, Decimal]:
    """Identity: log_{b}(x) = k * log_{b^k}(x)  <=> log_{b^k}(x) = log_b(x) / k"""
    lhs = log(base, x)
    rhs = to_decimal(k) * log(decimal_pow(base, k), x)
    return lhs, rhs


# Convenience utility to compare two Decimal values within current precision
def almost_equal(a: Decimal, b: Decimal) -> bool:
    """Return True if a and b are equal within the current decimal context.

    Uses a simple relative comparison based on context precision.
    """
    a_d = to_decimal(a)
    b_d = to_decimal(b)
    # If both are exactly equal, quick True
    if a_d == b_d:
        return True
    # Calculate a relative tolerance from precision: 10**(-prec+2)
    tol = Decimal(10) ** (Decimal(-_ctx.prec + 2))
    # Use relative diff
    try:
        diff = (a_d - b_d).copy_abs()
        denom = max(a_d.copy_abs(), b_d.copy_abs(), Decimal(1))
        return diff <= tol * denom
    except InvalidOperation:
        return False
