"""
trig.py

Complete Numerical & Symbolic Trigonometric/Hyperbolic Engine.
Imports STRICTLY TaylorSeries and DualMath from derivative.py.
Combines all classic formulas and residual identity functions without prefixes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from derivative import TaylorSeries, DualMath


# ==============================================================================
# CONFIGURATION & RADIANS CONVERSION ENGINE
# ==============================================================================

DEGREE_MODE = False  # Set to True if degree inputs are required


def to_radians(x):
    """Converts input to radians without external libraries."""
    if isinstance(x, str):
        s = x.strip().replace(" ", "").replace("pi", "π")
        if "π" in s:
            if s == "π":
                return DualMath.PI
            parts = s.split("π")
            num = float(parts[0]) if parts[0] not in ["", "+"] else (-1.0 if parts[0] == "-" else 1.0)
            den = float(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("/") else 1.0
            return (num * DualMath.PI) / den
        x = float(s)

    val = float(x)
    if DEGREE_MODE:
        return val * (DualMath.PI / 180.0)
    return val


def from_radians(x):
    """Converts radians back to degrees if DEGREE_MODE is active."""
    val = float(x)
    if DEGREE_MODE:
        return val * (180.0 / DualMath.PI)
    return val


# ==============================================================================
# TAYLOR SERIES NUMERIC EVALUATOR
# ==============================================================================

def eval_taylor_trig(fn_name, x, order=10):
    """
    Evaluates trigonometric and hyperbolic functions using DualMath 
    and TaylorSeries from derivative.py.
    """
    fn_map = {
        'sin': DualMath.sin,
        'cos': DualMath.cos,
        'tan': DualMath.tan,
        'arctan': DualMath.arctan,
        'arcsin': DualMath.arcsin,
        'arccos': DualMath.arccos,
        'sinh': DualMath.sinh,
        'cosh': DualMath.cosh,
        'tanh': DualMath.tanh,
    }
    if fn_name not in fn_map:
        raise ValueError(f"Function '{fn_name}' is not supported by DualMath Engine.")
    
    ts = TaylorSeries.seed(float(x), order=order)
    res_ts = fn_map[fn_name](ts)
    return res_ts.coeffs[0]


# ==============================================================================
# CORE & RECIPROCAL TRIGONOMETRIC FUNCTIONS
# ==============================================================================

def sin(x):
    val = to_radians(x)
    return eval_taylor_trig('sin', val)


def cos(x):
    val = to_radians(x)
    return eval_taylor_trig('cos', val)


def tan(x):
    val = to_radians(x)
    return eval_taylor_trig('tan', val)


def sec(x):
    return 1.0 / cos(x)


def csc(x):
    return 1.0 / sin(x)


def cot(x):
    return 1.0 / tan(x)


def asin(x):
    res = eval_taylor_trig('arcsin', float(x))
    return from_radians(res)


def acos(x):
    res = eval_taylor_trig('arccos', float(x))
    return from_radians(res)


def atan(x):
    res = eval_taylor_trig('arctan', float(x))
    return from_radians(res)


def asec(x):
    return acos(1.0 / float(x))


def acsc(x):
    return asin(1.0 / float(x))


def acot(x):
    return atan(1.0 / float(x))


# ==============================================================================
# HYPERBOLIC & INVERSE HYPERBOLIC FUNCTIONS
# ==============================================================================

def sinh(x):
    return eval_taylor_trig('sinh', float(x))


def cosh(x):
    return eval_taylor_trig('cosh', float(x))


def tanh(x):
    return eval_taylor_trig('tanh', float(x))


def sech(x):
    return 1.0 / cosh(x)


def csch(x):
    return 1.0 / sinh(x)


def coth(x):
    return 1.0 / tanh(x)


def asinh(x):
    val = float(x)
    return DualMath._ln_scalar(val + (val**2 + 1.0)**0.5)


def acosh(x):
    val = float(x)
    return DualMath._ln_scalar(val + (val**2 - 1.0)**0.5)


def atanh(x):
    val = float(x)
    return 0.5 * DualMath._ln_scalar((1.0 + val) / (1.0 - val))


# ==============================================================================
# CLASSIC EVALUATION IDENTITIES (PREFIX REMOVED)
# ==============================================================================

def pythagorean_1(x=0.5): return sin(x)**2 + cos(x)**2
def pythagorean_2(x=0.5): return 1.0 + tan(x)**2
def pythagorean_3(x=0.5): return 1.0 + cot(x)**2
def double_angle_sin_val(x=0.5): return 2.0 * sin(x) * cos(x)
def double_angle_cos_1_val(x=0.5): return cos(x)**2 - sin(x)**2
def double_angle_cos_2_val(x=0.5): return 2.0 * (cos(x)**2) - 1.0
def double_angle_cos_3_val(x=0.5): return 1.0 - 2.0 * (sin(x)**2)
def double_angle_tan_val(x=0.5): return (2.0 * tan(x)) / (1.0 - tan(x)**2)
def half_angle_sin_val(x=0.5): return (1.0 - cos(x)) / 2.0
def half_angle_cos_val(x=0.5): return (1.0 + cos(x)) / 2.0
def angle_sum_sin(a=0.5, b=0.3): return sin(a) * cos(b) + cos(a) * sin(b)
def angle_diff_sin(a=0.5, b=0.3): return sin(a) * cos(b) - cos(a) * sin(b)
def angle_sum_cos(a=0.5, b=0.3): return cos(a) * cos(b) - sin(a) * sin(b)
def angle_diff_cos(a=0.5, b=0.3): return cos(a) * cos(b) + sin(a) * sin(b)
def angle_sum_tan(a=0.5, b=0.3): return (tan(a) + tan(b)) / (1.0 - tan(a) * tan(b))
def sum_to_product_sin_sin(a=0.5, b=0.3): return 2.0 * sin((a + b) / 2.0) * cos((a - b) / 2.0)
def sum_to_product_sin_sub_val(a=0.5, b=0.3): return 2.0 * cos((a + b) / 2.0) * sin((a - b) / 2.0)
def sum_to_product_cos_cos(a=0.5, b=0.3): return 2.0 * cos((a + b) / 2.0) * cos((a - b) / 2.0)
def product_to_sum_sin_cos(a=0.5, b=0.3): return sin(a + b) + sin(a - b)
def cofunction_sin_val(x=0.5): return sin(DualMath.PI / 2.0 - float(x))
def cofunction_cos_val(x=0.5): return cos(DualMath.PI / 2.0 - float(x))
def even_odd_sin(x=0.5): return sin(-float(x))
def even_odd_cos(x=0.5): return cos(-float(x))
def triple_angle_sin_val(x=0.5): return 3.0 * sin(x) - 4.0 * (sin(x)**3)
def triple_angle_cos_val(x=0.5): return 4.0 * (cos(x)**3) - 3.0 * cos(x)
def triple_angle_tan_val(x=0.5): t = tan(x); return (3.0 * t - t**3) / (1.0 - 3.0 * (t**2))
def half_angle_tan_val(x=0.5): return (1.0 - cos(x)) / (1.0 + cos(x))
def angle_diff_tan(a=0.5, b=0.3): return (tan(a) - tan(b)) / (1.0 + tan(a) * tan(b))
def sum_to_product_cos_sub_val(a=0.5, b=0.3): return -2.0 * sin((a + b) / 2.0) * sin((a - b) / 2.0)
def product_to_sum_cos_sin(a=0.5, b=0.3): return sin(a + b) - sin(a - b)
def product_to_sum_cos_cos(a=0.5, b=0.3): return cos(a + b) + cos(a - b)
def product_to_sum_sin_sin(a=0.5, b=0.3): return cos(a + b) - cos(a - b)
def even_odd_tan(x=0.5): return tan(-float(x))
def cofunction_tan_val(x=0.5): return tan(DualMath.PI / 2.0 - float(x))
def cofunction_sec_val(x=0.5): return sec(DualMath.PI / 2.0 - float(x))
def cofunction_csc_val(x=0.5): return csc(DualMath.PI / 2.0 - float(x))
def cofunction_cot_val(x=0.5): return cot(DualMath.PI / 2.0 - float(x))
def power_reduction_sin_val(x=0.5): return (1.0 - cos(2.0 * float(x))) / 2.0
def power_reduction_cos_val(x=0.5): return (1.0 + cos(2.0 * float(x))) / 2.0
def power_reduction_tan_val(x=0.5): c2 = cos(2.0 * float(x)); return (1.0 - c2) / (1.0 + c2)
def bhagavata_sum(a=0.5, b=0.3): return sin(a)**2 - sin(b)**2
def bhagavata_sum_cos(a=0.5, b=0.3): return cos(a)**2 - sin(b)**2
def quadruple_angle_sin(x=0.5): s = sin(x); c = cos(x); return 4.0 * s * c * (1.0 - 2.0 * (s**2))
def quadruple_angle_cos(x=0.5): c = cos(x); return 8.0 * (c**4) - 8.0 * (c**2) + 1.0


# ==============================================================================
# NUMERICAL RESIDUAL IDENTITIES
# Returns difference from expected value (~0.0 indicates identity holds)
# ==============================================================================

# --- Pythagorean & Reciprocal ---
def pythagorean_sin_cos(x): return sin(x)**2 + cos(x)**2 - 1.0
def pythagorean_tan_sec(x): return 1.0 + tan(x)**2 - sec(x)**2
def pythagorean_cot_csc(x): return 1.0 + cot(x)**2 - csc(x)**2
def reciprocal_sec(x): return sec(x) - 1.0 / cos(x)
def reciprocal_csc(x): return csc(x) - 1.0 / sin(x)
def reciprocal_cot(x): return cot(x) - 1.0 / tan(x)
def quotient_tan(x): return tan(x) - (sin(x) / cos(x))
def quotient_cot(x): return cot(x) - (cos(x) / sin(x))

# --- Double-Angle ---
def double_angle_sin(x): return sin(2*x) - (2 * sin(x) * cos(x))
def double_angle_cos_1(x): return cos(2*x) - (cos(x)**2 - sin(x)**2)
def double_angle_cos_2(x): return cos(2*x) - (2*cos(x)**2 - 1.0)
def double_angle_cos_3(x): return cos(2*x) - (1.0 - 2*sin(x)**2)
def double_angle_tan(x): return tan(2*x) - ((2*tan(x)) / (1.0 - tan(x)**2))

# --- Half-Angle ---
def half_angle_sin(x): return sin(x / 2.0)**2 - ((1.0 - cos(x)) / 2.0)
def half_angle_cos(x): return cos(x / 2.0)**2 - ((1.0 + cos(x)) / 2.0)
def half_angle_tan_1(x): return tan(x / 2.0) - (sin(x) / (1.0 + cos(x)))
def half_angle_tan_2(x): return tan(x / 2.0) - ((1.0 - cos(x)) / sin(x))

# --- Angle Sum & Difference ---
def sum_sin(a, b): return sin(a + b) - (sin(a)*cos(b) + cos(a)*sin(b))
def diff_sin(a, b): return sin(a - b) - (sin(a)*cos(b) - cos(a)*sin(b))
def sum_cos(a, b): return cos(a + b) - (cos(a)*cos(b) - sin(a)*sin(b))
def diff_cos(a, b): return cos(a - b) - (cos(a)*cos(b) + sin(a)*sin(b))
def sum_tan(a, b): return tan(a + b) - ((tan(a) + tan(b)) / (1.0 - tan(a)*tan(b)))
def diff_tan(a, b): return tan(a - b) - ((tan(a) - tan(b)) / (1.0 + tan(a)*tan(b)))

# --- Even/Odd & Cofunction ---
def even_cos(x): return cos(-x) - cos(x)
def odd_sin(x): return sin(-x) + sin(x)
def odd_tan(x): return tan(-x) + tan(x)
def cofunction_sin(x): return sin(DualMath.PI/2.0 - to_radians(x)) - cos(x)
def cofunction_cos(x): return cos(DualMath.PI/2.0 - to_radians(x)) - sin(x)
def cofunction_tan(x): return tan(DualMath.PI/2.0 - to_radians(x)) - cot(x)

# --- Triple & Multiple Angle ---
def triple_angle_sin(x): return sin(3*x) - (3*sin(x) - 4*sin(x)**3)
def triple_angle_cos(x): return cos(3*x) - (4*cos(x)**3 - 3*cos(x))
def triple_angle_tan(x): return tan(3*x) - ((3*tan(x) - tan(x)**3) / (1.0 - 3*tan(x)**2))
def quad_angle_sin(x): return sin(4*x) - (4*sin(x)*cos(x)*(1.0 - 2*sin(x)**2))
def quad_angle_cos(x): return cos(4*x) - (8*cos(x)**4 - 8*cos(x)**2 + 1.0)
def quint_angle_sin(x): return sin(5*x) - (16*sin(x)**5 - 20*sin(x)**3 + 5*sin(x))
def quint_angle_cos(x): return cos(5*x) - (16*cos(x)**5 - 20*cos(x)**3 + 5*cos(x))

# --- Sum-to-Product & Product-to-Sum ---
def sum_to_product_sin_add(a, b): return (sin(a) + sin(b)) - (2 * sin((a+b)/2.0) * cos((a-b)/2.0))
def sum_to_product_sin_sub(a, b): return (sin(a) - sin(b)) - (2 * cos((a+b)/2.0) * sin((a-b)/2.0))
def sum_to_product_cos_add(a, b): return (cos(a) + cos(b)) - (2 * cos((a+b)/2.0) * cos((a-b)/2.0))
def sum_to_product_cos_sub(a, b): return (cos(a) - cos(b)) - (-2 * sin((a+b)/2.0) * sin((a-b)/2.0))
def prod_to_sum_sin_cos(a, b): return (2 * sin(a) * cos(b)) - (sin(a+b) + sin(a-b))
def prod_to_sum_cos_cos(a, b): return (2 * cos(a) * cos(b)) - (cos(a+b) + cos(a-b))
def prod_to_sum_sin_sin(a, b): return (-2 * sin(a) * sin(b)) - (cos(a+b) - cos(a-b))

# --- Power Reduction & Weierstrass ---
def power_reduction_sin(x): return sin(x)**2 - ((1.0 - cos(2*x)) / 2.0)
def power_reduction_cos(x): return cos(x)**2 - ((1.0 + cos(2*x)) / 2.0)
def weierstrass_sin(t): return (2*t / (1.0 + t**2))  # where t = tan(x/2)
def weierstrass_cos(t): return ((1.0 - t**2) / (1.0 + t**2))
def weierstrass_tan(t): return (2*t / (1.0 - t**2))

# --- Inverse Trigonometric Identities ---
def inverse_comp_asin_acos(x): return to_radians(asin(x)) + to_radians(acos(x)) - (DualMath.PI / 2.0)
def inverse_comp_atan_acot(x): return to_radians(atan(x)) + to_radians(acot(x)) - (DualMath.PI / 2.0)
def inverse_sum_atan(a, b): return to_radians(atan(a)) + to_radians(atan(b)) - to_radians(atan((a + b) / (1.0 - a*b)))
def inverse_diff_atan(a, b): return to_radians(atan(a)) - to_radians(atan(b)) - to_radians(atan((a - b) / (1.0 + a*b)))

# --- Hyperbolic Basic & Pythagorean ---
def hyperbolic_def_sinh(x): return sinh(x) - ((DualMath._exp_scalar(x) - DualMath._exp_scalar(-x)) / 2.0)
def hyperbolic_def_cosh(x): return cosh(x) - ((DualMath._exp_scalar(x) + DualMath._exp_scalar(-x)) / 2.0)
def hyperbolic_pythagorean(x): return cosh(x)**2 - sinh(x)**2 - 1.0
def hyperbolic_tanh_sech(x): return 1.0 - tanh(x)**2 - sech(x)**2
def hyperbolic_coth_csch(x): return coth(x)**2 - 1.0 - csch(x)**2

# --- Hyperbolic Double Angle & Addition ---
def hyperbolic_double_sinh(x): return sinh(2*x) - (2 * sinh(x) * cosh(x))
def hyperbolic_double_cosh(x): return cosh(2*x) - (cosh(x)**2 + sinh(x)**2)
def hyperbolic_sum_sinh(a, b): return sinh(a + b) - (sinh(a)*cosh(b) + cosh(a)*sinh(b))
def hyperbolic_sum_cosh(a, b): return cosh(a + b) - (cosh(a)*cosh(b) + sinh(a)*sinh(b))

# --- Inverse Hyperbolic & Classic Formulas ---
def asinh_log_def(x): return asinh(x) - DualMath._ln_scalar(x + (x**2 + 1.0)**0.5)
def acosh_log_def(x): return acosh(x) - DualMath._ln_scalar(x + (x**2 - 1.0)**0.5)
def machin_pi_formula(): return (4 * to_radians(atan(1.0/5.0)) - to_radians(atan(1.0/239.0))) - (DualMath.PI / 4.0)

