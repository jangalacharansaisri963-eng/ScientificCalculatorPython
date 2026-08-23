"""
Equation parser and equation solving router.
Parses strings with '=' and dispatches to linear, quadratic, system, or numerical root solvers.
"""
from typing import Any
import re
import math
from parser import parse_and_eval

def solve_quadratic(a, b, c):
    """Solve ax^2 + bx + c = 0. Returns a result dict."""
    import cmath, math
    disc = b*b - 4*a*c
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return {"type": "identity" if abs(c) < 1e-15 else "no_solution", "solution": None}
        return {"type": "unique_solution", "solution": -c/b, "roots": [-c/b]}
    if disc > 0:
        s = math.sqrt(disc)
        r1, r2 = (-b + s)/(2*a), (-b - s)/(2*a)
        return {"type": "two_real_roots", "roots": [r1, r2], "solution": r1}
    if abs(disc) < 1e-12:
        r = -b/(2*a)
        return {"type": "repeated_root", "roots": [r], "solution": r}
    s = cmath.sqrt(disc)
    r1, r2 = (-b + s)/(2*a), (-b - s)/(2*a)
    return {"type": "complex_roots", "roots": [r1, r2], "solution": r1}

def solve_linear_1var(b, c):
    """Solve b*x + c = 0."""
    if abs(b) < 1e-15:
        return {"type": "identity" if abs(c) < 1e-15 else "no_solution", "solution": None}
    return {"type": "unique_solution", "solution": -c/b, "roots": [-c/b]}

def solve_linear_system_2x2(a1, b1, c1, a2, b2, c2):
    """Solve a1*x + b1*y = c1, a2*x + b2*y = c2."""
    det = a1*b2 - a2*b1
    if abs(det) < 1e-15:
        return {"type": "no_unique_solution", "solution": None}
    x = (c1*b2 - c2*b1) / det
    y = (a1*c2 - a2*c1) / det
    return {"type": "unique_solution", "solution": {"x": x, "y": y}}


def _extract_poly_coeffs(expr_str: str, var_name: str = "x") -> tuple[float, float, float]:
    """
    Extract a, b, c from a quadratic / linear polynomial expression ax^2 + bx + c.
    Evaluates at x = 0, 1, -1 to find exact coefficients using Lagrange / finite differences.
    """
    # f(x) = a*x^2 + b*x + c
    # f(0) = c
    # f(1) = a + b + c
    # f(-1) = a - b + c
    # => a = (f(1) + f(-1) - 2*f(0)) / 2
    # => b = (f(1) - f(-1)) / 2
    # => c = f(0)
    try:
        f0 = float(parse_and_eval(expr_str, {var_name: 0.0}))
        f1 = float(parse_and_eval(expr_str, {var_name: 1.0}))
        fm1 = float(parse_and_eval(expr_str, {var_name: -1.0}))
        
        a = (f1 + fm1 - 2.0 * f0) / 2.0
        b = (f1 - fm1) / 2.0
        c = f0
        
        # Verify quadratic fit at x = 2
        f2 = float(parse_and_eval(expr_str, {var_name: 2.0}))
        expected_f2 = a * 4.0 + b * 2.0 + c
        if abs(f2 - expected_f2) > 1e-4:
            # Not a simple quadratic/linear polynomial
            return None, None, None
            
        return round(a, 9), round(b, 9), round(c, 9)
    except Exception:
        return None, None, None

def solve_equation(equation_str: str, var_name: str = "x") -> dict[str, Any]:
    """
    Parse and solve an equation. Supports:
    1. 'LHS = RHS'
    2. 'f(x)' (assumed equal to 0)
    """
    eq = equation_str.strip()
    if '=' in eq:
        parts = eq.split('=', 1)
        lhs, rhs = parts[0].strip(), parts[1].strip()
        diff_expr = f"({lhs}) - ({rhs})"
    else:
        diff_expr = eq

    # Try extracting quadratic/linear coefficients
    a, b, c = _extract_poly_coeffs(diff_expr, var_name)
    
    if a is not None and b is not None and c is not None:
        if abs(a) > 1e-9:
            # Quadratic
            quad_res = solve_quadratic(a, b, c)
            quad_res["var"] = var_name
            quad_res["equation"] = equation_str
            return quad_res
        elif abs(b) > 1e-9 or abs(c) > 1e-9:
            # Linear in 1 variable: b*x + c = 0
            lin_res = solve_linear_1var(b, c)
            lin_res["var"] = var_name
            lin_res["equation"] = equation_str
            return lin_res
        else:
            return {"type": "identity", "message": "Identity (0 = 0)", "solution": None}
            
    # If not polynomial, use Newton-Raphson numerical solver
    try:
        def f(val):
            return float(parse_and_eval(diff_expr, {var_name: val}))
            
        x_curr = 1.0
        for _ in range(50):
            fx = f(x_curr)
            if abs(fx) < 1e-12:
                break
            h = 1e-6
            dfx = (f(x_curr + h) - f(x_curr - h)) / (2 * h)
            if abs(dfx) < 1e-14:
                break
            x_curr = x_curr - fx / dfx
            
        return {
            "type": "numerical_root",
            "var": var_name,
            "solution": x_curr,
            "residual": abs(f(x_curr)),
            "steps": [f"Numerically solved {diff_expr} = 0 via Newton-Raphson: {var_name} ≈ {x_curr}"]
        }
    except Exception as err:
        return {"type": "error", "error": str(err)}
