"""
compare.py

Comparison and ordering functions.
"""


MAX_VALUES = 10


def _check(values, name):
    """
    Validates the number of arguments.
    """

    if not 2 <= len(values) <= MAX_VALUES:
        raise ValueError(
            f"{name} requires 2 to {MAX_VALUES} values."
        )


# ==========================================
# BASIC COMPARISON
# ==========================================

def compare(a, b):
    """
    Compare two values.

    Returns:
        "<" if a < b
        ">" if a > b
        "=" if a == b
    """

    if a == b:
        return "="

    return "<" if a < b else ">"


def compare3(a, b):
    """
    Three-way comparison.

    Returns:
        -1 if a < b
         0 if a == b
         1 if a > b
    """

    if a < b:
        return -1

    if a > b:
        return 1

    return 0


# ==========================================
# MULTI-VALUE COMPARISON
# ==========================================

def less(*values):
    """
    Returns True if:
    a < b < c < ...
    """

    _check(values, "less")

    return all(
        values[i] < values[i + 1]
        for i in range(len(values) - 1)
    )


def greater(*values):
    """
    Returns True if:
    a > b > c > ...
    """

    _check(values, "greater")

    return all(
        values[i] > values[i + 1]
        for i in range(len(values) - 1)
    )


def equal(*values):
    """
    Returns True if all values are equal.
    """

    _check(values, "equal")

    return all(
        value == values[0]
        for value in values
    )


def not_equal(*values):
    """
    Returns True if every value is unique.
    """

    _check(values, "not_equal")

    return len(set(values)) == len(values)


def less_equal(*values):
    """
    Returns True if:
    a <= b <= c <= ...
    """

    _check(values, "less_equal")

    return all(
        values[i] <= values[i + 1]
        for i in range(len(values) - 1)
    )


def greater_equal(*values):
    """
    Returns True if:
    a >= b >= c >= ...
    """

    _check(values, "greater_equal")

    return all(
        values[i] >= values[i + 1]
        for i in range(len(values) - 1)
    )


# ==========================================
# ORDERING
# ==========================================

def AO(*values):
    """
    Ascending Order.

    Returns a sorted list.
    """

    _check(values, "AO")

    return sorted(values)


def DO(*values):
    """
    Descending Order.

    Returns a sorted list.
    """

    _check(values, "DO")

    return sorted(
        values,
        reverse=True,
    )


# ==========================================
# EXTREMES
# ==========================================

def greatest(*values):
    """
    Returns the greatest value.
    """

    _check(values, "greatest")

    return max(values)


def least(*values):
    """
    Returns the least value.
    """

    _check(values, "least")

    return min(values)
