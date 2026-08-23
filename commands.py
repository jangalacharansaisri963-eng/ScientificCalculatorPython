"""
CLI command dispatch and help documentation for interactive REPL.
"""
from typing import Any
import sys
from constants import CONSTANTS_DICT
from functions import FUNCTION_REGISTRY, load_all

HELP_TEXT = """
======================= ScientificCalculator =======================
Interactive Python Scientific Calculator & Math Toolkit

Core Features & Syntax:
  - Arithmetic & Power:   2 + 3 * 4, 2^8, 10 % 3, 5!
  - Trigonometry:         sin(pi/2), cos(0), tan(pi/4), asin(1)
  - Hyperbolic:           sinh(1), cosh(0), atanh(0.5)
  - Logarithms:           ln(e), log(100), log2(64), exp(2)
  - Roots:                sqrt(144), cbrt(27), nroot(32, 5)
  - Combinatorics:        nCr(10, 3), nPr(6, 2), fact(5)
  - Number Theory:        gcd(48, 18), lcm(12, 15), is_prime(97)
  - Complex Numbers:      3 + 4j, (2+3j) * (1-2j), sqrt(-16)
  - Variables:            x = 25, y = x * 2, ans + 10
  - Equation Solver:      solve 2*x + 5 = 15, solve x^2 - 5*x + 6 = 0
  - Physics Calculations: ke(m=2, v=10), coulomb(q1=1e-6, q2=2e-6, r=0.05)

REPL Commands:
  :help                 Show this help menu
  :vars                 List stored user variables
  :consts               List available physical and mathematical constants
  :funcs [category]     List supported math functions
  :mode [RAD|DEG]       Switch angle mode (current: RAD)
  :solve <equation>     Solve algebraic or linear equation
  :history              Show recent calculation history
  :clear                Clear calculation history and variables
  :test                 Run all internal unit tests
  :quit or :exit        Exit calculator
====================================================================
"""

def handle_command(cmd_line: str, engine: Any) -> tuple[bool, str]:
    """
    Handle a REPL special command starting with ':' or keyword.
    Returns (handled: bool, output_message: str).
    """
    cmd = cmd_line.strip()
    if not cmd.startswith(':') and not cmd.lower().startswith('solve '):
        return False, ""

    if cmd.lower().startswith('solve '):
        eq = cmd[6:].strip()
        res = engine.evaluate(eq)
        return True, str(res.get("formatted", res))

    parts = cmd[1:].strip().split(maxsplit=1)
    base_cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if base_cmd in ('help', 'h', '?'):
        return True, HELP_TEXT

    elif base_cmd in ('vars', 'var'):
        if not engine.variables:
            return True, "No user variables set."
        out = ["Defined variables:"]
        for k, v in engine.variables.items():
            out.append(f"  {k} = {v}")
        return True, "\n".join(out)

    elif base_cmd in ('consts', 'constants'):
        out = ["Available Constants:"]
        for k, v in sorted(CONSTANTS_DICT.items()):
            out.append(f"  {k:<10} = {v}")
        return True, "\n".join(out)

    elif base_cmd in ('funcs', 'functions'):
        out = ["Available Functions:"]
        categories = {}
        load_all()
        for fname, info in FUNCTION_REGISTRY.items():
            cat = info.get("category", "General")
            categories.setdefault(cat, []).append(f"{fname} ({info.get('desc', '')})")
        for cat, flist in sorted(categories.items()):
            out.append(f"\n[{cat}]")
            for item in flist:
                out.append(f"  • {item}")
        return True, "\n".join(out)

    elif base_cmd == 'mode':
        if not arg:
            return True, f"Current angle mode: {engine.get_angle_mode()}"
        msg = engine.set_angle_mode(arg)
        return True, msg

    elif base_cmd == 'history':
        if not engine.history:
            return True, "History is empty."
        out = ["Calculation History:"]
        for idx, h in enumerate(engine.history, 1):
            out.append(f" {idx:2d}. {h['expression']} => {h['formatted']}")
        return True, "\n".join(out)

    elif base_cmd == 'clear':
        engine.clear_history()
        engine.clear_variables()
        return True, "History and variables cleared."

    elif base_cmd in ('quit', 'exit', 'q'):
        return True, "__QUIT__"

    return False, f"Unknown command ':{base_cmd}'. Type :help for commands."
