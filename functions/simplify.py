"""
simplify.py

Simplifies numeric values and expressions.
"""

from fractions import Fraction


def repeating_decimal(value):
    """
    Converts repeating decimals.

    Examples:
    0.(3) -> 1/3
    1.2(3) -> 37/30
    """

    if "(" not in value:
        return None

    before, repeat = value.split("(")
    repeat = repeat.replace(")", "")

    if "." not in before:
        return None

    whole, decimal = before.split(".")

    numerator = (
        int(whole + decimal + repeat)
        -
        int(whole + decimal)
    )

    denominator = (
        (10 ** len(decimal))
        *
        (10 ** len(repeat) - 1)
    )

    return Fraction(
        numerator,
        denominator
    ).limit_denominator()


def simplify(value):
    """
    Simplify numbers and expressions.
    """

    # Fraction
    if isinstance(value, Fraction):
        return value.limit_denominator()

    # Integer
    if isinstance(value, int):
        return value

    # Float
    if isinstance(value, float):

        fraction = Fraction(
            value
        ).limit_denominator()

        if fraction.denominator == 1:
            return fraction.numerator()

        return fraction

    # Complex
    if isinstance(value, complex):
        return value

    # Other numeric values
    if not isinstance(value, str):

        try:

            fraction = Fraction(
                value
            ).limit_denominator()

            if fraction.denominator == 1:
                return fraction.numerator

            return fraction

        except Exception:

            return value

    # String input
    value = value.strip()

    # Repeating decimals
    repeat = repeating_decimal(value)

    if repeat is not None:
        return repeat

    # Evaluate expressions
    try:

        from functions import FUNCTION_REGISTRY

        scope = {
            "__builtins__": None,
        }
        for name, meta in FUNCTION_REGISTRY.items():
            scope[name] = meta["func"]

        result = eval(
            value,
            scope
        )

        return simplify(result)

    except Exception:
        pass

    # Plain fraction string
    try:

        fraction = Fraction(
            value
        ).limit_denominator()

        if fraction.denominator == 1:
            return fraction.numerator

        return fraction

    except Exception:

        return value
