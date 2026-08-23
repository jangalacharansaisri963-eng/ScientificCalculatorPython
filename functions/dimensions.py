"""
Dimensional Analysis and Physical Quantity Dimensions System.
Supports fundamental dimensions:
M: Mass
L: Length
T: Time
I: Electric Current (or A)
K: Temperature (or Theta)
N: Amount of substance (mol)
J: Luminous intensity (cd)

Loads a comprehensive dataset of physical quantities from `ScientificCalculator/data/dimensions.json`.
Supports multiplication (*), division (/), powers (^), and parsing expressions of physical quantities
like 'Pressure/Force' -> [L^-2], 'Force * Distance' -> [M L^2 T^-2] (Energy), etc.
"""
from typing import Dict, Union, Optional, Any
import fractions
import json
import os
import re

class Dimension:
    """
    Represents a physical dimension as powers of base physical quantities:
    M (Mass), L (Length), T (Time), I (Current), K (Temperature), N (Amount), J (Luminous Intensity)
    """
    BASE_SYMBOLS = ("M", "L", "T", "I", "K", "N", "J")
    
    def __init__(self,
                 M: Union[int, float, fractions.Fraction] = 0,
                 L: Union[int, float, fractions.Fraction] = 0,
                 T: Union[int, float, fractions.Fraction] = 0,
                 I: Union[int, float, fractions.Fraction] = 0,
                 K: Union[int, float, fractions.Fraction] = 0,
                 N: Union[int, float, fractions.Fraction] = 0,
                 J: Union[int, float, fractions.Fraction] = 0,
                 name: Optional[str] = None):
        self.powers: Dict[str, fractions.Fraction] = {
            "M": fractions.Fraction(M),
            "L": fractions.Fraction(L),
            "T": fractions.Fraction(T),
            "I": fractions.Fraction(I),
            "K": fractions.Fraction(K),
            "N": fractions.Fraction(N),
            "J": fractions.Fraction(J),
        }
        self.name = name

    def is_dimensionless(self) -> bool:
        return all(p == 0 for p in self.powers.values())

    def __mul__(self, other: Union['Dimension', int, float]) -> 'Dimension':
        if isinstance(other, (int, float)):
            # Scalar multiplication doesn't change dimensions
            return Dimension(**{k: v for k, v in self.powers.items()})
        if isinstance(other, Dimension):
            new_powers = {k: self.powers[k] + other.powers[k] for k in self.BASE_SYMBOLS}
            return Dimension(**new_powers)
        return NotImplemented

    def __rmul__(self, other: Union['Dimension', int, float]) -> 'Dimension':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Dimension', int, float]) -> 'Dimension':
        if isinstance(other, (int, float)):
            return Dimension(**{k: v for k, v in self.powers.items()})
        if isinstance(other, Dimension):
            new_powers = {k: self.powers[k] - other.powers[k] for k in self.BASE_SYMBOLS}
            return Dimension(**new_powers)
        return NotImplemented

    def __rtruediv__(self, other: Union['Dimension', int, float]) -> 'Dimension':
        if isinstance(other, (int, float)):
            # 1 / Dimension -> inverted powers
            new_powers = {k: -self.powers[k] for k in self.BASE_SYMBOLS}
            return Dimension(**new_powers)
        return NotImplemented

    def __pow__(self, power: Union[int, float, fractions.Fraction]) -> 'Dimension':
        p = fractions.Fraction(power)
        new_powers = {k: self.powers[k] * p for k in self.BASE_SYMBOLS}
        return Dimension(**new_powers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return all(self.powers[k] == other.powers[k] for k in self.BASE_SYMBOLS)

    def format_power(self, power: fractions.Fraction) -> str:
        if power.denominator == 1:
            return str(power.numerator)
        return f"{power.numerator}/{power.denominator}"

    def to_formula_string(self) -> str:
        """
        Returns string representation like '[M L^-1 T^-2]' or 'Dimensionless (1)'.
        """
        if self.is_dimensionless():
            return "[1] (Dimensionless)"
        
        parts = []
        for sym in self.BASE_SYMBOLS:
            p = self.powers[sym]
            if p != 0:
                if p == 1:
                    parts.append(sym)
                else:
                    parts.append(f"{sym}^{self.format_power(p)}")
        return f"[{' '.join(parts)}]"

    def __repr__(self) -> str:
        return f"Dimension({self.to_formula_string()})"

    def __str__(self) -> str:
        return self.to_formula_string()

    def identify_quantity(self) -> Optional[str]:
        """
        Identifies if this dimensional formula matches a known named physical quantity.
        """
        for name, dim in KNOWN_QUANTITIES.items():
            if self == dim and name.lower() not in ("dimensionless", "1"):
                return name.replace("_", " ").title()
        return None


# Global registry of quantities and their metadata
KNOWN_QUANTITIES: Dict[str, Dimension] = {}
QUANTITY_METADATA: Dict[str, Dict[str, Any]] = {}

def load_dimensions_dataset(json_path: Optional[str] = None):
    """
    Loads dimensional formulas from the JSON dataset file.
    """
    global KNOWN_QUANTITIES, QUANTITY_METADATA
    
    if json_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "data", "dimensions.json")
        
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                quantities = data.get("quantities", {})
                for name, info in quantities.items():
                    dim = Dimension(
                        M=info.get("M", 0),
                        L=info.get("L", 0),
                        T=info.get("T", 0),
                        I=info.get("I", 0),
                        K=info.get("K", 0),
                        N=info.get("N", 0),
                        J=info.get("J", 0),
                        name=name
                    )
                    KNOWN_QUANTITIES[name.lower()] = dim
                    QUANTITY_METADATA[name.lower()] = info
                    
                    # Also register variants
                    KNOWN_QUANTITIES[name.replace("_", "")] = dim
                    KNOWN_QUANTITIES[name.upper()] = dim
                    KNOWN_QUANTITIES[name.capitalize()] = dim
        except Exception as e:
            print(f"Warning: Failed to load dimensions dataset JSON: {e}")

# Initial load on module import
load_dimensions_dataset()

def get_dimension_of(name: str) -> Optional[Dimension]:
    """Resolves a string name to its physical Dimension."""
    cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in KNOWN_QUANTITIES:
        return KNOWN_QUANTITIES[cleaned]
    # Check base symbols directly M, L, T, I, K, N, J
    if name.strip().upper() in Dimension.BASE_SYMBOLS:
        return Dimension(**{name.strip().upper(): 1})
    return None

def is_dimension_query(expr: str) -> bool:
    """
    Determines if an expression contains physical quantities intended for dimensional analysis.
    E.g. 'Pressure/Force', 'Force * Distance', 'dim(Energy)', 'Velocity^2 / Acceleration'
    """
    s = expr.strip()
    if s.lower().startswith("dim(") and s.endswith(")"):
        return True
    
    # Check if the tokens in expression contain known physical quantities
    words = re.findall(r'[a-zA-Z_]+', s)
    if not words:
        return False
    
    for w in words:
        if get_dimension_of(w) is not None:
            return True
            
    return False

def evaluate_dimensional_expression(expr: str) -> dict:
    """
    Evaluates an algebraic combination of physical quantities with multiplication,
    division, and powers.
    Example: 'Pressure/Force' -> Dimension: [L^-2], Matched Quantity: None, Formula: [L^-2]
    """
    s = expr.strip()
    if s.lower().startswith("dim(") and s.endswith(")"):
        s = s[4:-1].strip()

    # Tokenizer for dimensional expressions
    token_pattern = r'([a-zA-Z_]+|\d+(?:\.\d+)?|\*\*|\^|\*|\/|\(|\))'
    tokens = re.findall(token_pattern, s)
    
    if not tokens:
        raise ValueError(f"Empty or invalid dimensional expression: '{expr}'")

    output_queue = []
    operator_stack = []
    
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '**': 3}
    associativity = {'*': 'L', '/': 'L', '^': 'R', '**': 'R'}

    for token in tokens:
        # Number (exponent or scalar)
        if re.match(r'^\d+(\.\d+)?$', token):
            output_queue.append(float(token) if '.' in token else int(token))
            
        # Identifier (Physical Quantity or Base Symbol M, L, T, etc.)
        elif re.match(r'^[a-zA-Z_]+$', token):
            dim = get_dimension_of(token)
            if dim is None:
                raise ValueError(f"Unknown physical quantity or dimension: '{token}'")
            output_queue.append(dim)
            
        elif token in ('*', '/', '^', '**'):
            op = '^' if token == '**' else token
            while operator_stack:
                top = operator_stack[-1]
                if top in ('*', '/', '^'):
                    prec1 = precedence[op]
                    prec2 = precedence[top]
                    assoc = associativity[op]
                    if (assoc == 'L' and prec1 <= prec2) or (assoc == 'R' and prec1 < prec2):
                        output_queue.append(operator_stack.pop())
                    else:
                        break
                else:
                    break
            operator_stack.append(op)
            
        elif token == '(':
            operator_stack.append(token)
            
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if not operator_stack:
                raise ValueError("Mismatched parentheses in dimensional expression")
            operator_stack.pop() # Pop '('
            
    while operator_stack:
        top = operator_stack.pop()
        if top in ('(', ')'):
            raise ValueError("Mismatched parentheses in dimensional expression")
        output_queue.append(top)

    # Evaluate RPN
    stack = []
    for item in output_queue:
        if isinstance(item, (Dimension, int, float)):
            stack.append(item)
        elif item in ('*', '/', '^'):
            if len(stack) < 2:
                raise ValueError(f"Insufficient operands for operator '{item}'")
            b = stack.pop()
            a = stack.pop()
            
            if item == '*':
                if isinstance(a, Dimension) and isinstance(b, Dimension):
                    stack.append(a * b)
                elif isinstance(a, Dimension) and isinstance(b, (int, float)):
                    stack.append(a * b)
                elif isinstance(a, (int, float)) and isinstance(b, Dimension):
                    stack.append(b * a)
                else:
                    stack.append(a * b)
                    
            elif item == '/':
                if isinstance(a, Dimension) and isinstance(b, Dimension):
                    stack.append(a / b)
                elif isinstance(a, Dimension) and isinstance(b, (int, float)):
                    stack.append(a / b)
                elif isinstance(a, (int, float)) and isinstance(b, Dimension):
                    stack.append(a / b)
                else:
                    stack.append(a / b)
                    
            elif item == '^':
                if isinstance(a, Dimension) and isinstance(b, (int, float)):
                    stack.append(a ** b)
                else:
                    raise ValueError(f"Dimensional power requires base Dimension and numeric exponent, got {type(a)} ^ {type(b)}")

    if len(stack) != 1:
        raise ValueError(f"Malformed dimensional expression evaluation")

    result_dim = stack[0]
    if not isinstance(result_dim, Dimension):
        result_dim = Dimension()

    formula_str = result_dim.to_formula_string()
    matching_name = result_dim.identify_quantity()
    
    return {
        "expression": expr,
        "dimensional_formula": formula_str,
        "powers": {k: float(v) if v.denominator != 1 else int(v) for k, v in result_dim.powers.items()},
        "matching_quantity": matching_name,
        "is_dimensionless": result_dim.is_dimensionless(),
        "formatted": f"{formula_str}" + (f" ({matching_name})" if matching_name else "")
    }

def list_known_quantities() -> list[dict]:
    """Returns all physical quantities in the dataset with their units and formulas."""
    results = []
    for name, info in QUANTITY_METADATA.items():
        dim = KNOWN_QUANTITIES[name]
        results.append({
            "name": name,
            "formula": dim.to_formula_string(),
            "unit": info.get("unit", ""),
            "desc": info.get("desc", "")
        })
    return results

