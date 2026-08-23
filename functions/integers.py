"""
integers.py

Integer mathematics.
"""

import math


def gcd(*numbers):

    if len(numbers) < 2:
        raise ValueError(
            "At least 2 numbers are required."
        )

    if len(numbers) > 10:
        raise ValueError(
            "Maximum 10 numbers allowed."
        )

    integers = []

    for n in numbers:

        if n != int(n):
            raise ValueError(
                "GCD/HCF requires whole numbers."
            )

        integers.append(int(n))

    result = integers[0]

    for n in integers[1:]:

        result = math.gcd(result, n)

    return result


def lcm(*numbers):

    if len(numbers) < 2:
        raise ValueError(
            "At least 2 numbers are required."
        )

    if len(numbers) > 10:
        raise ValueError(
            "Maximum 10 numbers allowed."
        )

    integers = []

    for n in numbers:

        if n != int(n):
            raise ValueError(
                "LCM requires whole numbers."
            )

        integers.append(int(n))

    result = integers[0]

    for n in integers[1:]:

        result = abs(result * n) // math.gcd(result, n)

    return result
