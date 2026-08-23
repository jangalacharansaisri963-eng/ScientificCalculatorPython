"""
roots.py

Root functions.
"""

from decimal import Decimal
import math


def sqrt(x):
    return Decimal(str(x)).sqrt()


def cbrt(x):
    return Decimal(str(x)) ** (Decimal(1) / Decimal(3))


def root(n, x):
    """
    nth root of x.

    root(2, 25) -> 5
    root(3, 64) -> 4
    """

    return Decimal(str(x)) ** (Decimal(1) / Decimal(n))


def sqrtrem(x):
    """
    Integer square root and remainder.

    sqrtrem(26)
    -> Answer: 5, Remainder: 1
    """

    value = int(x)

    root = math.isqrt(value)

    remainder = value - (root * root)

    return f"Answer: {root}, Remainder: {remainder}"


def cbrtrem(x):
    """
    Integer cube root and remainder.

    cbrtrem(30)
    -> Answer: 3, Remainder: 3
    """

    value = int(x)

    root = int(round(value ** (1 / 3)))

    while (root + 1) ** 3 <= value:
        root += 1

    while root ** 3 > value:
        root -= 1

    remainder = value - (root ** 3)

    return f"Answer: {root}, Remainder: {remainder}"


def nextsquare(x):
    """
    Returns the next perfect square
    greater than or equal to x.
    """

    value = int(x)

    root = math.isqrt(value)

    if root * root == value:
        return value

    return (root + 1) ** 2


def prevsquare(x):
    """
    Returns the previous perfect square
    less than or equal to x.
    """

    value = int(x)

    root = math.isqrt(value)

    return root ** 2


def nextcube(x):
    """
    Returns the next perfect cube
    greater than or equal to x.
    """

    value = int(x)

    root = int(round(value ** (1 / 3)))

    while root ** 3 < value:
        root += 1

    while (root - 1) ** 3 >= value:
        root -= 1

    return root ** 3


def prevcube(x):
    """
    Returns the previous perfect cube
    less than or equal to x.
    """

    value = int(x)

    root = int(round(value ** (1 / 3)))

    while root ** 3 > value:
        root -= 1

    while (root + 1) ** 3 <= value:
        root += 1

    return root ** 3


def isperfectsquare(x):
    """
    Returns information about whether
    a number is a perfect square.
    """

    value = int(x)

    root = math.isqrt(value)

    if root * root == value:

        return (
            f"{value} IS a perfect square.\n"
            f"Square Root: {root}"
        )

    lower = root * root
    upper = (root + 1) ** 2

    return (
        f"{value} is NOT a perfect square.\n\n"
        f"Nearest lower perfect square:\n"
        f"{lower} (subtract {value - lower})\n\n"
        f"Nearest higher perfect square:\n"
        f"{upper} (add {upper - value})"
    )


def isperfectcube(x):
    """
    Returns information about whether
    a number is a perfect cube.
    """

    value = int(x)

    root = int(round(value ** (1 / 3)))

    while root ** 3 > value:
        root -= 1

    while (root + 1) ** 3 <= value:
        root += 1

    if root ** 3 == value:

        return (
            f"{value} IS a perfect cube.\n"
            f"Cube Root: {root}"
        )

    lower = root ** 3
    upper = (root + 1) ** 3

    return (
        f"{value} is NOT a perfect cube.\n\n"
        f"Nearest lower perfect cube:\n"
        f"{lower} (subtract {value - lower})\n\n"
        f"Nearest higher perfect cube:\n"
        f"{upper} (add {upper - value})"
    )
