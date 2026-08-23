"""
Formatting and presentation functions for calculator results.
"""
from typing import Any
import math
from utils.decimal_display import (
    to_significant_figures,
    to_engineering_notation,
    to_scientific_notation,
    float_to_fraction_str,
    to_base_n_str
)

def format_number(val: Any, precision: int = 8) -> str:
    """Format float, int, complex, or other values cleanly."""
    if isinstance(val, complex):
        re = val.real
        im = val.imag
        if abs(re) < 1e-14:
            re = 0.0
        if abs(im) < 1e-14:
            im = 0.0
        
        if im == 0:
            return format_number(re, precision)
        if re == 0:
            return f"{format_number(im, precision)}j"
        
        sign = "+" if im >= 0 else "-"
        return f"({format_number(re, precision)} {sign} {format_number(abs(im), precision)}j)"
    
    if isinstance(val, (int, bool)):
        return str(val)
    
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        if abs(val) < 1e-14:
            return "0"
        if val.is_integer() and abs(val) < 1e12:
            return str(int(val))
        
        # Check if very close to integer
        if abs(val - round(val)) < 1e-11:
            return str(int(round(val)))
            
        return to_significant_figures(val, precision)
    
    if isinstance(val, list):
        # Vector or matrix
        if all(isinstance(row, list) for row in val):
            return format_matrix(val)
        return "[" + ", ".join(format_number(x, precision) for x in val) + "]"
    
    if isinstance(val, dict):
        return str(val)
    
    return str(val)

def format_matrix(mat: list[list[Any]]) -> str:
    """Format 2D matrix in aligned rows."""
    if not mat:
        return "[]"
    lines = []
    str_grid = [[format_number(cell, 4) for cell in row] for row in mat]
    col_widths = [max(len(str_grid[r][c]) for r in range(len(mat))) for c in range(len(mat[0]))]
    
    for row in str_grid:
        formatted_row = "  ".join(cell.rjust(col_widths[c]) for c, cell in enumerate(row))
        lines.append(f"[ {formatted_row} ]")
    return "\n".join(lines)

def format_detailed_result(expr: str, val: Any) -> dict[str, Any]:
    """Return dictionary with multiple representations."""
    formatted = format_number(val)
    res: dict[str, Any] = {
        "expression": expr,
        "raw": val,
        "formatted": formatted,
        "type": type(val).__name__
    }
    
    if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
        num = float(val)
        res["scientific"] = to_scientific_notation(num)
        res["engineering"] = to_engineering_notation(num)
        res["fraction"] = float_to_fraction_str(num)
        if num.is_integer() and -2**31 <= int(num) <= 2**31:
            i_val = int(num)
            res["binary"] = bin(i_val)
            res["hex"] = hex(i_val)
            res["octal"] = oct(i_val)
    elif isinstance(val, complex):
        res["real"] = val.real
        res["imag"] = val.imag
        res["magnitude"] = abs(val)
        res["phase_rad"] = math.atan2(val.imag, val.real)
        res["phase_deg"] = math.degrees(math.atan2(val.imag, val.real))
        
    return res
