"""
math_custom.py

A pure Python mathematics engine built entirely from scratch.

Features
--------
• Decimal high precision
• Custom mathematical constants
• Exponential functions
• Logarithms
• Trigonometric functions
• Inverse trigonometric functions

No function from Python's math module is used.
"""

from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    getcontext,
    localcontext
)

# ============================================================
# DECIMAL PRECISION
# ============================================================

DEFAULT_PRECISION = 512

getcontext().prec = DEFAULT_PRECISION


# ============================================================
# HIGH PRECISION CONSTANTS
# ============================================================

_PI_STR = (
    "3.14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
    "82148086513282306647093844609550582231725359408128"
    "48111745028410270193852110555964462294895493038196"
)

_E_STR = (
    "2.71828182845904523536028747135266249775724709369995"
    "95749669676277240766303535475945713821785251664274"
    "27466391932003059921817413596629043572900334295260"
    "59563073813232862794349076323382988075319525101901"
)


# ============================================================
# CONSTANTS
# ============================================================

CUSTOM_INF = Decimal("Infinity")

CUSTOM_NAN = Decimal("NaN")


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _epsilon():
    """
    Internal convergence tolerance.
    """

    return Decimal(10) ** (
        -(getcontext().prec - 5)
    )


# ============================================================
# PRECISION CONTROL
# ============================================================

def set_precision(digits):
    """
    Changes Decimal precision.
    """

    digits = int(digits)

    if digits < 10:
        raise ValueError(
            "Precision must be at least 10."
        )

    getcontext().prec = digits


def get_precision():
    """
    Returns current precision.
    """

    return getcontext().prec


# ============================================================
# CONSTANT ACCESSORS
# ============================================================

def CUSTOM_PI(digits=None):
    """
    Returns π rounded to the requested
    number of decimal places.

    Maximum: 500 digits.
    """

    if digits is None:
        return Decimal(_PI_STR)

    digits = int(digits)

    if digits < 1 or digits > 500:
        raise ValueError(
            "Digits must be between 1 and 500."
        )

    with localcontext() as ctx:

        ctx.prec = digits + 2

        value = Decimal(_PI_STR)

        quantizer = Decimal(
            "1." + ("0" * digits)
        )

        return value.quantize(quantizer)


def CUSTOM_E(digits=None):
    """
    Returns Euler's number rounded
    to the requested precision.
    """

    if digits is None:
        return Decimal(_E_STR)

    digits = int(digits)

    if digits < 1 or digits > 500:
        raise ValueError(
            "Digits must be between 1 and 500."
        )

    with localcontext() as ctx:

        ctx.prec = digits + 2

        value = Decimal(_E_STR)

        quantizer = Decimal(
            "1." + ("0" * digits)
        )

        return value.quantize(quantizer)


CUSTOM_TAU = CUSTOM_PI() * Decimal(2)


# ============================================================
# BASIC UTILITIES
# ============================================================

def custom_abs(x):
    """
    Absolute value.
    """

    x = Decimal(str(x))

    if x < 0:
        return -x

    return x


def custom_floor(x):
    """
    Floor.
    """

    return int(
        Decimal(str(x)).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )


def custom_ceil(x):
    """
    Ceiling.
    """

    return int(
        Decimal(str(x)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

# ============================================================
# FACTORIAL
# ============================================================

def custom_factorial(n):
    """
    Computes n! iteratively.
    """

    n = int(n)

    if n < 0:
        raise ValueError(
            "Factorial is undefined for negative numbers."
        )

    result = 1

    for i in range(2, n + 1):
        result *= i

    return result

# ==============================================================================
# GAMMA FUNCTION & GENERALIZED FACTORIAL
# ==============================================================================

def custom_gamma(x):
    """
    Computes the Gamma function Γ(x) using the Lanczos approximation.

    Properties:
        Γ(n) = (n - 1)!       for positive integers n
        Γ(x + 1) = xΓ(x)
        Γ(1) = 1
        Γ(1/2) = √π

    Supports positive and negative non-integer values through
    the reflection formula.

    No math/scipy imports are used.
    """

    z = Decimal(str(x))

    # Gamma has poles at 0, -1, -2, -3, ...
    if z == 0:
        raise ValueError("Gamma function undefined at 0.")

    # Check negative integers.
    if z < 0:
        if z == z.to_integral_value():
            raise ValueError(
                "Gamma function undefined at non-positive integers."
            )

        # Reflection formula:
        # Γ(z)Γ(1-z) = π / sin(πz)
        sin_term = custom_sin(CUSTOM_PI() * z)

        if sin_term == 0:
            raise ValueError("Gamma function undefined at this value.")

        return CUSTOM_PI() / (sin_term * custom_gamma(Decimal(1) - z))

    # Lanczos approximation coefficients.
    coefficients = (
        Decimal("0.99999999999980993"),
        Decimal("676.5203681218851"),
        Decimal("-1259.1392167224028"),
        Decimal("771.32342877765313"),
        Decimal("-176.61502916214059"),
        Decimal("12.507343278686905"),
        Decimal("-0.13857109526572012"),
        Decimal("0.0000099843695780195716"),
        Decimal("0.00000015056327351493116"),
    )

    # Lanczos approximation works with z >= 1/2 after shifting.
    if z < Decimal("0.5"):
        sin_term = custom_sin(CUSTOM_PI() * z)

        if sin_term == 0:
            raise ValueError("Gamma function undefined at this value.")

        return CUSTOM_PI() / (
            sin_term * custom_gamma(Decimal(1) - z)
        )

    z_minus_one = z - Decimal(1)

    total = coefficients[0]

    for i in range(1, len(coefficients)):
        total += coefficients[i] / (
            z_minus_one + Decimal(i)
        )

    g = Decimal(7)

    t = z_minus_one + g + Decimal("0.5")

    sqrt_two_pi = custom_sqrt(
        Decimal(2) * CUSTOM_PI()
    )

    result = (
        sqrt_two_pi
        * custom_pow(t, z_minus_one + Decimal("0.5"))
        * custom_exp(-t)
        * total
    )

    return +result


def custom_gamma_factorial(n):
    """
    Computes the generalized factorial using the Gamma function.

    Mathematical definition:
        n! = Γ(n + 1)

    Therefore:

        custom_gamma_factorial(0) = 1
        custom_gamma_factorial(1) = 1
        custom_gamma_factorial(2) = 2
        custom_gamma_factorial(3) = 6
        custom_gamma_factorial(5) = 120

    It also works for non-integer values:

        custom_gamma_factorial(0.5) = Γ(1.5)
        custom_gamma_factorial(2.5) = Γ(3.5)

    No math/scipy imports are used.
    """

    d = Decimal(str(n))

    # Generalized factorial Γ(n + 1) has poles at
    # n = -1, -2, -3, ...
    if d < 0 and d == d.to_integral_value():
        raise ValueError(
            "Factorial is undefined for negative integers."
        )

    return custom_gamma(d + Decimal(1))


# ============================================================
# GREATEST COMMON DIVISOR
# ============================================================

def custom_gcd(a, b):
    """
    Euclidean algorithm.
    """

    a = abs(int(a))
    b = abs(int(b))

    while b:
        a, b = b, a % b

    return a


# ============================================================
# SQUARE ROOT
# ============================================================

def custom_sqrt(x):
    """
    Computes square root using
    Newton's method.
    """

    x = Decimal(str(x))

    if x < 0:
        raise ValueError(
            "Cannot compute square root of a negative number."
        )

    if x == 0:
        return Decimal(0)

    guess = x / 2

    eps = _epsilon()

    while True:

        next_guess = (
            guess +
            x / guess
        ) / 2

        if custom_abs(
            next_guess - guess
        ) < eps:
            return +next_guess

        guess = next_guess


# ============================================================
# EXPONENTIAL
# ============================================================

def custom_exp(x):
    """
    Computes e^x using
    Taylor series.
    """

    x = Decimal(str(x))

    term = Decimal(1)
    total = Decimal(1)

    n = 1

    eps = _epsilon()

    while True:

        term *= x
        term /= Decimal(n)

        total += term

        if custom_abs(term) < eps:
            break

        n += 1

    return +total


def W(x, branch=0):
    """
    Computes the Lambert W function without using the math module.

    Lambert W is defined by:

        W(x) * exp(W(x)) = x

    Parameters
    ----------
    x : number
        Input value.
    branch : int
        Lambert W branch.

        0  -> principal branch W_0
        -1 -> lower real branch W_{-1}

    Returns
    -------
    Decimal

    Real-valued domain:
        W_0  : x >= -1/e
        W_-1 : -1/e <= x < 0

    Examples
    --------
    W(0)       -> 0
    W(1)       -> approximately 0.5671432904
    W(2)       -> approximately 0.8526055020
    W(-1/e)   -> -1
    """

    x = Decimal(str(x))

    if branch not in (0, -1):
        raise ValueError(
            "Only real Lambert W branches 0 and -1 are supported."
        )

    # e^-1 = 1/e
    negative_limit = -Decimal(1) / CUSTOM_E()

    # ------------------------------------------------------------------
    # Domain checking
    # ------------------------------------------------------------------

    if branch == 0:

        if x < negative_limit:
            raise ValueError(
                "W(x) is not real for x < -1/e on the principal branch."
            )

    else:

        if x < negative_limit or x >= 0:
            raise ValueError(
                "The real W_-1 branch requires -1/e <= x < 0."
            )

    # ------------------------------------------------------------------
    # Special cases
    # ------------------------------------------------------------------

    if x == 0:
        if branch == 0:
            return Decimal(0)

    if x == negative_limit:
        return Decimal(-1)

    # ------------------------------------------------------------------
    # Initial approximation
    # ------------------------------------------------------------------

    if branch == 0:

        if x < Decimal("1"):
            w = x
        else:
            # log-based starting approximation
            lx = custom_ln(x)
            llx = custom_ln(lx)

            w = lx - llx

    else:

        # W_-1 is <= -1
        lx = custom_ln(-x)

        if x > Decimal("-0.1"):
            w = lx - custom_ln(-lx)
        else:
            w = lx - custom_ln(-lx)

    # ------------------------------------------------------------------
    # Halley iteration
    #
    # W_{n+1} =
    #
    # w - (w*e^w - x) /
    #     (e^w*(w+1) - ((w+2)*(w*e^w-x))/(2*w+2))
    #
    # ------------------------------------------------------------------

    precision = getcontext().prec

    tolerance = Decimal(10) ** (-(precision - 8))

    max_iterations = 100

    for _ in range(max_iterations):

        ew = custom_exp(w)

        f = w * ew - x

        wp1 = w + Decimal(1)

        denominator = (
            ew * wp1
            - (
                (w + Decimal(2))
                * f
                / (Decimal(2) * wp1)
            )
        )

        if denominator == 0:
            break

        next_w = w - f / denominator

        if custom_abs(next_w - w) < tolerance:
            return +next_w

        w = next_w

    return +w


# ============================================================
# NATURAL LOGARITHM
# ============================================================

def custom_ln(x):
    """
    Computes natural logarithm.
    """

    x = Decimal(str(x))

    if x <= 0:
        raise ValueError(
            "Math domain error."
        )

    one = Decimal(1)
    two = Decimal(2)

    k = 0

    while x > two:
        x /= two
        k += 1

    while x < one / two:
        x *= two
        k -= 1

    y = (x - one) / (x + one)

    y2 = y * y

    term = y
    total = y

    n = 3

    eps = _epsilon()

    while True:

        term *= y2

        add = term / Decimal(n)

        total += add

        if custom_abs(add) < eps:
            break

        n += 2

    LN2 = Decimal(
        "0.6931471805599453094172321214581765680755001343602552"
    )

    return +(two * total + Decimal(k) * LN2)


def custom_ln2():
    """
    Computes ln(2) dynamically to the current Decimal precision.

    Uses the rapidly converging identity:
        ln(2) = 2 * arctanh(1/3)

    which expands to:
        ln(2) = 2 * (1/3 + 1/(3^3*3) + 1/(3^5*5) + ...)
    """

    current_prec = getcontext().prec

    cache_key = ("ln2", current_prec)
    if cache_key in _CONSTANT_CACHE:
        return _CONSTANT_CACHE[cache_key]

    with localcontext() as ctx:
        # Extra guard digits for intermediate calculations
        ctx.prec = current_prec + 5

        one = Decimal(1)
        two = Decimal(2)
        three = Decimal(3)

        y = one / three
        y2 = y * y

        term = y
        total = Decimal(0)
        n = 1

        # Convergence threshold
        eps = one.scaleb(-ctx.prec)

        while True:
            add = term / n
            total += add

            if custom_abs(add) < eps:
                break

            term *= y2
            n += 2

        ln2 = +(two * total)

    _CONSTANT_CACHE[cache_key] = ln2
    return ln2


# ============================================================
# LOGARITHM
# ============================================================

def custom_log(x, base=None):
    """
    Computes logarithm of x.

    If base is omitted,
    natural logarithm is returned.
    """

    if base is None:
        return custom_ln(x)

    base = Decimal(str(base))

    if base <= 0:
        raise ValueError(
            "Base must be positive."
        )

    if base == 1:
        raise ValueError(
            "Base cannot equal one."
        )

    return (
        custom_ln(x)
        /
        custom_ln(base)
    )


# ============================================================
# POWER
# ============================================================

def custom_pow(x, y):
    """
    Computes x^y.
    """

    x = Decimal(str(x))
    y = Decimal(str(y))

    if x == 0:

        if y <= 0:
            raise ValueError(
                "Undefined power."
            )

        return Decimal(0)

    if x < 0:

        if y != int(y):
            raise ValueError(
                "Negative base requires an integer exponent."
            )

        exponent = int(y)

        answer = custom_exp(
            Decimal(exponent)
            *
            custom_ln(-x)
        )

        if exponent % 2:
            return -answer

        return answer

    return custom_exp(
        y *
        custom_ln(x)
    )

# ============================================================
# SINE
# ============================================================

def custom_sin(x):
    """
    Computes sine using
    the Taylor series.

    Input must be in radians.
    """

    x = Decimal(str(x))

    two_pi = CUSTOM_TAU

    # Better argument reduction
    x = x % two_pi

    if x > CUSTOM_PI():
        x -= two_pi

    term = x
    total = x

    x2 = x * x

    n = 1

    eps = _epsilon()

    while True:

        term *= -x2
        term /= Decimal(
            (2 * n) * (2 * n + 1)
        )

        total += term

        if custom_abs(term) < eps:
            break

        n += 1

    return +total


# ============================================================
# COSINE
# ============================================================

def custom_cos(x):
    """
    Computes cosine using
    the Taylor series.

    Input must be in radians.
    """

    x = Decimal(str(x))

    two_pi = CUSTOM_TAU

    x = x % two_pi

    if x > CUSTOM_PI():
        x -= two_pi

    term = Decimal(1)
    total = Decimal(1)

    x2 = x * x

    n = 1

    eps = _epsilon()

    while True:

        term *= -x2
        term /= Decimal(
            (2 * n - 1) *
            (2 * n)
        )

        total += term

        if custom_abs(term) < eps:
            break

        n += 1

    return +total


# ============================================================
# TANGENT
# ============================================================

def custom_tan(x):
    """
    Computes tangent.
    """

    c = custom_cos(x)

    if custom_abs(c) < _epsilon():
        raise ZeroDivisionError(
            "Tangent undefined."
        )

    return custom_sin(x) / c


# ============================================================
# SECANT
# ============================================================

def custom_sec(x):
    """
    Computes secant.
    """

    c = custom_cos(x)

    if custom_abs(c) < _epsilon():
        raise ZeroDivisionError(
            "Secant undefined."
        )

    return Decimal(1) / c


# ============================================================
# COSECANT
# ============================================================

def custom_csc(x):
    """
    Computes cosecant.
    """

    s = custom_sin(x)

    if custom_abs(s) < _epsilon():
        raise ZeroDivisionError(
            "Cosecant undefined."
        )

    return Decimal(1) / s


# ============================================================
# COTANGENT
# ============================================================

def custom_cot(x):
    """
    Computes cotangent.
    """

    s = custom_sin(x)

    if custom_abs(s) < _epsilon():
        raise ZeroDivisionError(
            "Cotangent undefined."
        )

    return custom_cos(x) / s


# ============================================================
# ANGLE CONVERSION
# ============================================================

def custom_radians(degrees):
    """
    Degrees → Radians.
    """

    degrees = Decimal(str(degrees))

    return (
        degrees *
        CUSTOM_PI()
        /
        Decimal(180)
    )


def custom_degrees(radians):
    """
    Radians → Degrees.
    """

    radians = Decimal(str(radians))

    return (
        radians *
        Decimal(180)
        /
        CUSTOM_PI()
    )

# ============================================================
# INVERSE TANGENT
# ============================================================

def custom_atan(x):
    """
    Computes inverse tangent.

    Returns radians.
    """

    x = Decimal(str(x))

    one = Decimal(1)

    if x == 0:
        return Decimal(0)

    if x < 0:
        return -custom_atan(-x)

    if x > one:
        return (
            CUSTOM_PI() / 2
            -
            custom_atan(one / x)
        )

    term = x
    total = x

    x2 = x * x

    n = 1

    eps = _epsilon()

    while True:

        term *= -x2

        add = term / Decimal(
            2 * n + 1
        )

        total += add

        if custom_abs(add) < eps:
            break

        n += 1

    return +total


# ============================================================
# INVERSE TANGENT (TWO ARGUMENTS)
# ============================================================

def custom_atan2(y, x):
    """
    Computes atan2(y, x).

    Returns radians.
    """

    y = Decimal(str(y))
    x = Decimal(str(x))

    if x > 0:
        return custom_atan(y / x)

    if x < 0 and y >= 0:
        return (
            custom_atan(y / x)
            +
            CUSTOM_PI()
        )

    if x < 0 and y < 0:
        return (
            custom_atan(y / x)
            -
            CUSTOM_PI()
        )

    if x == 0 and y > 0:
        return CUSTOM_PI() / 2

    if x == 0 and y < 0:
        return -CUSTOM_PI() / 2

    raise ValueError(
        "atan2(0, 0) is undefined."
    )


# ============================================================
# INVERSE SINE
# ============================================================

def custom_asin(x):
    """
    Computes inverse sine.

    Returns radians.
    """

    x = Decimal(str(x))

    if x < -1 or x > 1:
        raise ValueError(
            "Math domain error."
        )

    if x == 1:
        return CUSTOM_PI() / 2

    if x == -1:
        return -CUSTOM_PI() / 2

    return custom_atan(
        x
        /
        custom_sqrt(
            Decimal(1)
            -
            x * x
        )
    )


# ============================================================
# INVERSE COSINE
# ============================================================

def custom_acos(x):
    """
    Computes inverse cosine.

    Returns radians.
    """

    x = Decimal(str(x))

    if x < -1 or x > 1:
        raise ValueError(
            "Math domain error."
        )

    return (
        CUSTOM_PI() / 2
        -
        custom_asin(x)
    )

def custom_log10(x):
    """
    Base-10 logarithm.
    """
    return custom_log(x, Decimal(10))


def custom_log2(x):
    """
    Base-2 logarithm.
    """
    return custom_log(x, Decimal(2))


# ============================================================
# ALIASES
# ============================================================

custom_arcsin = custom_asin
custom_arccos = custom_acos
custom_arctan = custom_atan
