"""
factors.py

Factor and prime factorization functions.
"""

import math


def _to_integer(value):

    value = float(value)

    if not value.is_integer():
        raise ValueError("Input must be an integer.")

    return int(value)


def factors(n):

    n = _to_integer(n)

    if n == 0:
        return [0]

    n = abs(n)

    result = []

    limit = math.isqrt(n)

    for i in range(1, limit + 1):

        if n % i == 0:

            result.append(i)

            if i != n // i:
                result.append(n // i)

    return sorted(result)


def factorization(n):

    n = _to_integer(n)

    if n == 0:
        return "0"

    if n == 1:
        return "1"

    negative = n < 0

    n = abs(n)

    factors = []

    divisor = 2

    while divisor * divisor <= n:

        while n % divisor == 0:

            factors.append(str(divisor))

            n //= divisor

        divisor += 1

    if n > 1:
        factors.append(str(n))

    answer = " × ".join(factors)

    if negative:
        answer = "-1 × " + answer

    return answer
