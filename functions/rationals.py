"""
rationals.py

Find rational numbers between two values.
"""

import math
from fractions import Fraction
# Import the clean LINE variable from your formatter
LINE = "-" * 40  # local fallback


def fr(a, b, count):

    a = Fraction(str(a))
    b = Fraction(str(b))

    if a > b:
        a, b = b, a

    found = []

    denom = 1

    while len(found) < count:

        for numer in range(
            math.floor(float(a * denom)) + 1,
            math.ceil(float(b * denom))
        ):

            f = Fraction(numer, denom)

            if a < f < b and f not in found:

                found.append(f)

                if len(found) >= count:
                    break

        denom += 1

    found.sort()

    print()
    # Now using the standard dash LINE from formatter.py
    print(LINE)
    print(f"{count} Rational Numbers")
    print(f"Between {float(a)} and {float(b)}")
    print(LINE)

    for f in found:
        print(f"{f} = {float(f)}")

    print(LINE)
    
