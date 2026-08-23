"""
Core calculation engine for ScientificCalculator.
Manages evaluation context, variables, angle mode, history, and command dispatch.
"""
from typing import Any
import math
from parser import parse_and_eval, tokenize, parse_to_rpn, evaluate_rpn
from formatter import format_number, format_detailed_result
from equation_parser import solve_equation
from constants import CONSTANTS_DICT
from functions import FUNCTION_REGISTRY, load_all, get_function
from functions.dimensions import is_dimension_query, evaluate_dimensional_expression

class CalculatorEngine:
    """Stateful calculator engine."""
    def __init__(self, angle_mode: str = "RAD"):
        self.angle_mode = angle_mode.upper()  # "RAD" or "DEG"
        self.variables: dict[str, Any] = {
            "ans": 0.0
        }
        self.history: list[dict[str, Any]] = []
        self.max_history = 100
        load_all()  # populate function registry

    def set_angle_mode(self, mode: str) -> str:
        mode_up = mode.upper()
        if mode_up not in ("RAD", "DEG", "GRAD"):
            raise ValueError("Angle mode must be RAD, DEG, or GRAD")
        self.angle_mode = mode_up
        return f"Angle mode set to {self.angle_mode}"

    def get_angle_mode(self) -> str:
        return self.angle_mode

    def set_variable(self, name: str, value: Any) -> Any:
        if name in CONSTANTS_DICT:
            raise ValueError(f"Cannot overwrite protected constant '{name}'")
        self.variables[name] = value
        return value

    def get_variable(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        if name in CONSTANTS_DICT:
            return CONSTANTS_DICT[name]
        raise KeyError(f"Variable '{name}' not defined")

    def clear_variables(self) -> None:
        self.variables = {"ans": 0.0}

    def clear_history(self) -> None:
        self.history.clear()

    def evaluate(self, expr_or_stmt: str) -> dict[str, Any]:
        """
        Evaluate an expression, assignment (e.g. 'x = 10 + 2'), or command.
        Returns a rich result dictionary.
        """
        s = expr_or_stmt.strip()
        if not s:
            return {"status": "empty", "result": None}

        # Check variable assignment: e.g. 'x = 12 * 4'
        # But distinguish from equation solving '2*x + 4 = 10'
        assign_match = False
        if '=' in s:
            left_part, right_part = s.split('=', 1)
            left_clean = left_part.strip()
            # If left is a pure single identifier like 'x' or 'my_var'
            if left_clean.isidentifier() and left_clean not in CONSTANTS_DICT:
                try:
                    res_val = parse_and_eval(right_part, self.variables, self.angle_mode)
                    self.set_variable(left_clean, res_val)
                    self.variables["ans"] = res_val
                    formatted = format_number(res_val)
                    self._add_history(s, res_val, formatted)
                    return {
                        "status": "assignment",
                        "variable": left_clean,
                        "value": res_val,
                        "formatted": formatted
                    }
                except Exception:
                    # Fallback to equation solving if assignment fails
                    pass

        # Check equation solver syntax: contains '=' with variables on both or non-identifier left
        if '=' in s:
            sol = solve_equation(s)
            self._add_history(s, sol, str(sol))
            return {
                "status": "equation_solved",
                "solution": sol,
                "formatted": str(sol)
            }

        # Check dimensional formula analysis (e.g. 'Pressure/Force', 'Force*Distance', 'dim(Power)')
        if is_dimension_query(s):
            try:
                dim_res = evaluate_dimensional_expression(s)
                self._add_history(s, dim_res, dim_res["formatted"])
                return {
                    "status": "dimension",
                    "dimensional_formula": dim_res["dimensional_formula"],
                    "matching_quantity": dim_res["matching_quantity"],
                    "powers": dim_res["powers"],
                    "formatted": dim_res["formatted"]
                }
            except Exception as dim_err:
                # If dimensional parse was attempted and failed, or fallback to math eval
                pass

        # Standard mathematical expression evaluation
        try:
            val = parse_and_eval(s, self.variables, self.angle_mode)
            self.variables["ans"] = val
            formatted = format_number(val)
            detailed = format_detailed_result(s, val)
            self._add_history(s, val, formatted)
            return {
                "status": "success",
                "result": val,
                "formatted": formatted,
                "detailed": detailed
            }
        except Exception as err:
            return {
                "status": "error",
                "error": str(err),
                "expression": s
            }

    def _add_history(self, expr: str, result: Any, formatted: str) -> None:
        self.history.append({
            "expression": expr,
            "result": result,
            "formatted": formatted,
            "mode": self.angle_mode
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
