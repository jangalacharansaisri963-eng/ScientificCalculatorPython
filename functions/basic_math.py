"""
basic_math.py

Basic mathematical utility functions.
"""

from math import isqrt, factorial


def square(x):
    return x * x

def pow(x, y):
    return x**y

def power(x, y):
    """Alias for pow."""
    return x**y

def cube(x):
    return x * x * x


def reciprocal(x):
    if x == 0:
        raise ZeroDivisionError(
            "Cannot take the reciprocal of zero."
        )

    return 1 / x


def lerp(a, b, t):
    """
    Linear interpolation.

    t = 0 -> a
    t = 1 -> b
    """

    return a + (b - a) * t


def is_even(n):
    return int(n) % 2 == 0


def is_odd(n):
    return int(n) % 2 == 1


def is_prime(n):

    n = int(n)

    if n < 2:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    limit = isqrt(n)

    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False

    return True


def next_prime(n):

    n = int(n) + 1

    while not is_prime(n):
        n += 1

    return n


def previous_prime(n):

    n = int(n) - 1

    while n >= 2:

        if is_prime(n):
            return n

        n -= 1

    raise ValueError(
        "No previous prime exists."
    )


def prime_factors(n):

    n = int(n)

    if n < 2:
        return []

    factors = []

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    divisor = 3

    while divisor * divisor <= n:

        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor

        divisor += 2

    if n > 1:
        factors.append(n)

    return factors


def factor_count(n):

    n = abs(int(n))

    if n == 0:
        raise ValueError(
            "Zero has infinitely many factors."
        )

    count = 0

    limit = isqrt(n)

    for i in range(1, limit + 1):

        if n % i == 0:

            if i == n // i:
                count += 1
            else:
                count += 2

    return count


def digit_sum(n):

    digits = str(abs(int(n)))

    return sum(
        int(digit)
        for digit in digits
    )


def digit_product(n):

    digits = str(abs(int(n)))

    product = 1

    for digit in digits:
        product *= int(digit)

    return product


def reverse_number(n):

    negative = int(n) < 0

    reversed_number = int(
        str(abs(int(n)))[::-1]
    )

    if negative:
        reversed_number *= -1

    return reversed_number


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError(
            "Cannot divide by zero."
        )

    return a / b


def apsum(a, b):
    return (a + b) * b / 2


def apsub(a, b):
    return (a - b) * b / 2


def apterm(a, d, n):
    return a + (n - 1) * d


def gpsum(a, r, n):
    if r == 1:
        return a * n

    return a * (r ** n - 1) / (r - 1)


def gpterm(a, r, n):
    return a * r ** (n - 1)


def arithmean(a, b):
    return (a + b) / 2


def average(a, b):
    return (a + b) / 2


def percentage(value, percent):
    return value * percent / 100


def percentof(a, b):
    return (a / b) * 100


def increase(value, percent):
    return value * (1 + percent / 100)


def decrease(value, percent):
    return value * (1 - percent / 100)


def ratio(a, b):
    return a / b


def proportion(a, b, c):
    return (b * c) / a


def simple_interest(p, r, t):
    return (p * r * t) / 100


def amount(p, r, t):
    return p + simple_interest(p, r, t)


def compound_amount(p, r, n, t):
    return p * (1 + r / (100 * n)) ** (n * t)


def compound_interest(p, r, n, t):
    return compound_amount(p, r, n, t) - p


def triangular(n):
    return n * (n + 1) / 2


def nsum(n):
    return n * (n + 1) / 2


def squaresum(n):
    return n * (n + 1) * (2 * n + 1) / 6


def cubesum(n):
    return (n * (n + 1) / 2) ** 2


def oddsum(n):
    return n ** 2


def evensum(n):
    return n * (n + 1)


def sumofintegers(a, b):
    return (a + b) * (b - a + 1) / 2


def remainder(a, b):
    return a % b


def quotient(a, b):
    if b == 0:
        raise ZeroDivisionError(
            "Cannot divide by zero."
        )

    return a // b


def pythagoras(a, b):
    return (a ** 2 + b ** 2) ** 0.5


def hypotenuse(a, b):
    return (a ** 2 + b ** 2) ** 0.5

def abs_value(x):
    return abs(x)


def min_value(a, b):
    return min(a, b)


def max_value(a, b):
    return max(a, b)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def midpoint(a, b):
    return (a + b) / 2


def range_value(a, b):
    return abs(b - a)


def percent_change(old, new):
    if old == 0:
        raise ZeroDivisionError(
            "Cannot calculate percentage change from zero."
        )

    return ((new - old) / old) * 100


def ratio_sum(a, b):
    return a + b


def ratio_difference(a, b):
    return a - b


def direct_proportion(a, b, c):
    return (b * c) / a


def inverse_proportion(a, b, c):
    return (a * b) / c


def nth_term(a, d, n):
    return a + (n - 1) * d


def arithmetic_mean(*values):
    if not values:
        raise ValueError(
            "At least one value is required."
        )

    return sum(values) / len(values)


def geometric_mean(a, b):
    if a < 0 or b < 0:
        raise ValueError(
            "Geometric mean requires non-negative values."
        )

    return (a * b) ** 0.5


def harmonic_mean(a, b):
    if a == 0 or b == 0:
        raise ZeroDivisionError(
            "Cannot calculate harmonic mean with zero."
        )

    return 2 * a * b / (a + b)


def factorial_ratio(n, r):
    if r < 0 or n < 0 or r > n:
        raise ValueError(
            "Invalid n or r."
        )

    return factorial(n) / factorial(n - r)


def permutation(n, r):
    if r < 0 or n < 0 or r > n:
        raise ValueError(
            "Invalid n or r."
        )

    return factorial(n) / factorial(n - r)


def combination(n, r):
    if r < 0 or n < 0 or r > n:
        raise ValueError(
            "Invalid n or r."
        )

    return factorial(n) / (
        factorial(r) * factorial(n - r)
    )


def remainder_percentage(part, whole):
    if whole == 0:
        raise ZeroDivisionError(
            "Whole cannot be zero."
        )

    return (part % whole) / whole * 100
    


def modulo(a, b):
    """Modulo / remainder a % b."""
    if b == 0:
        raise ZeroDivisionError("Modulo by zero")
    return a % b
