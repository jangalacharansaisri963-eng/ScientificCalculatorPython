"""
quadratic.py

Quadratic equation utilities.

Equation:
    ax² + bx + c = 0
"""

import cmath


def discriminant(a, b, c):
    """
    Returns the discriminant.

    Δ = b² - 4ac
    """

    return b * b - 4 * a * c


def nature(a, b, c):
    """
    Returns the nature of the roots.
    """

    d = discriminant(a, b, c)

    if d > 0:
        return "Two distinct real roots"

    if d == 0:
        return "Two equal real roots"

    return "Two complex roots"


def repeated_root(a, b, c):
    """
    Returns the repeated root.

    Raises ValueError if roots are not equal.
    """

    if discriminant(a, b, c) != 0:
        raise ValueError(
            "Equation does not have repeated roots."
        )

    return -b / (2 * a)


def quadratic(a, b, c):
    """
    Returns both roots as a tuple.

    Example:
        quadratic(1,5,6)
        (-2.0, -3.0)
    """

    if a == 0:
        raise ValueError(
            "Coefficient 'a' cannot be zero."
        )

    d = discriminant(a, b, c)

    root = cmath.sqrt(d)

    x1 = (-b + root) / (2 * a)
    x2 = (-b - root) / (2 * a)

    # Convert purely real complex numbers
    if abs(x1.imag) < 1e-12:
        x1 = x1.real

    if abs(x2.imag) < 1e-12:
        x2 = x2.real

    return (x1, x2)
