"""
Decimal and number formatting utilities.
"""
from decimal import Decimal, getcontext
import math
from fractions import Fraction

getcontext().prec = 30

def to_significant_figures(value: float, sig_figs: int = 6) -> str:
    """Format a float to a specified number of significant figures."""
    if value == 0:
        return "0"
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    
    try:
        d = Decimal(str(value))
        return f"{float(d):.{sig_figs}g}"
    except Exception:
        return f"{value:.{sig_figs}g}"

def to_engineering_notation(value: float, precision: int = 4) -> str:
    """Format number into engineering notation (exponent is multiple of 3)."""
    if value == 0:
        return "0"
    if math.isnan(value) or math.isinf(value):
        return str(value)
    
    sign = "-" if value < 0 else ""
    val = abs(value)
    exp = int(math.floor(math.log10(val)))
    eng_exp = exp - (exp % 3)
    eng_mantissa = val / (10 ** eng_exp)
    
    return f"{sign}{eng_mantissa:.{precision}f}e{eng_exp:+03d}"

def to_scientific_notation(value: float, precision: int = 6) -> str:
    """Format number into standard scientific notation."""
    if math.isnan(value) or math.isinf(value):
        return str(value)
    return f"{value:.{precision}e}"

def float_to_fraction_str(value: float, max_denominator: int = 100000) -> str:
    """Convert float to exact or approximate fraction string."""
    if math.isnan(value) or math.isinf(value):
        return str(value)
    
    frac = Fraction(value).limit_denominator(max_denominator)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"

def to_base_n_str(value: int, base: int) -> str:
    """Convert integer to string representation in base 2..36."""
    if not isinstance(value, int):
        if float(value).is_integer():
            value = int(value)
        else:
            raise ValueError("Base conversions only apply to integers")
    
    if value == 0:
        return "0"
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if base < 2 or base > len(digits):
        raise ValueError(f"Base must be between 2 and {len(digits)}")
    
    sign = "-" if value < 0 else ""
    val = abs(value)
    result = []
    while val > 0:
        result.append(digits[val % base])
        val //= base
    return sign + "".join(reversed(result))
