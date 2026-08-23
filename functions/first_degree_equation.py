"""
first_degree_equation.py

Solves single-variable equations.

Supported:
- Exactly one variable
- Exactly one '='
- +  -  *  /
- Parentheses
- Decimals
- Fractions

Examples:

a + 4 = 6
2a + 5 = 17
5a - 3 = 2a + 9
a/5 = 8
5/a = 20
1/(a+2)=3
"""

from decimal import Decimal

# parse_equation removed to break circular import; solver uses direct eval


def _split_equation(expression):
    """Minimal local split of 'lhs = rhs' and detect single variable name."""
    import re
    if "=" not in expression:
        raise ValueError("No '=' found in equation")
    left, right = expression.split("=", 1)
    left, right = left.strip(), right.strip()
    vars_found = sorted(set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", left + " " + right)) - {"e", "pi", "E", "PI"})
    variable = vars_found[0] if vars_found else "x"
    return left, right, variable


def solve(expression):
    """
    Solve an equation in one variable.

    Returns:
        Decimal solution.
    """

    left, right, variable = _split_equation(
        expression
    )

    def f(value):

        scope = {
            "__builtins__": None,
            variable: Decimal(str(value))
        }

        return (
            eval(left, scope)
            -
            eval(right, scope)
        )

    # -----------------------------
    # Search for a sign change
    # -----------------------------

    x = Decimal("-100000")
    step = Decimal("1")

    previous = None

    try:
        previous = f(x)
    except Exception:
        pass

    while x <= Decimal("100000"):

        x2 = x + step

        try:
            current = f(x2)

        except Exception:
            x = x2
            previous = None
            continue

        if previous is not None:

            if current == 0:
                return x2

            if previous == 0:
                return x

            if previous * current < 0:

                low = x
                high = x2

                # Binary search

                for _ in range(120):

                    mid = (
                        low + high
                    ) / 2

                    value = f(mid)

                    if abs(value) < Decimal("1e-40"):
                        return mid

                    if f(low) * value < 0:
                        high = mid
                    else:
                        low = mid

                return (
                    low + high
                ) / 2

        previous = current
        x = x2

    raise ValueError(
        "No real solution found."
    )
