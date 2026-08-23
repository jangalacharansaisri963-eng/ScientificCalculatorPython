"""
hyperbolic.py

Hyperbolic functions.
"""

import math
from decimal import Decimal


def sinh(x):
    return Decimal(str(math.sinh(float(x))))


def cosh(x):
    return Decimal(str(math.cosh(float(x))))


def tanh(x):
    return Decimal(str(math.tanh(float(x))))
