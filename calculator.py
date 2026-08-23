"""
ScientificCalculator Main CLI Application & Entry Point.
"""
import sys
import os
import argparse
import json
import unittest
from engine import CalculatorEngine
from commands import handle_command, HELP_TEXT

class ScientificCalculatorCLI:
    def __init__(self, angle_mode: str = "RAD"):
        self.engine = CalculatorEngine(angle_mode=angle_mode)

    def run_eval(self, expression: str, output_json: bool = False) -> int:
        """Evaluate a single expression and print result."""
        res = self.engine.evaluate(expression)
        if output_json:
            print(json.dumps(res, indent=2, default=str))
        else:
            if res.get("status") == "error":
                print(f"Error: {res.get('error')}", file=sys.stderr)
                return 1
            elif res.get("status") == "assignment":
                print(f"{res['variable']} = {res['formatted']}")
            elif res.get("status") == "equation_solved":
                print(json.dumps(res['solution'], indent=2, default=str))
            else:
                print(res.get("formatted"))
        return 0

    def run_repl(self):
        """Interactive Read-Eval-Print-Loop."""
        print("=" * 60)
        print(" Scientific Calculator (Python 3)")
        print(" Type :help for commands, :mode DEG to switch modes, or :exit to quit")
        print("=" * 60)
        
        while True:
            try:
                mode_indicator = f"[{self.engine.get_angle_mode()}]"
                user_input = input(f"{mode_indicator} >> ").strip()
                if not user_input:
                    continue

                if user_input.startswith(':') or user_input.lower().startswith('solve '):
                    handled, msg = handle_command(user_input, self.engine)
                    if handled:
                        if msg == "__QUIT__":
                            print("Goodbye!")
                            break
                        print(msg)
                        continue

                # Normal evaluation
                res = self.engine.evaluate(user_input)
                if res.get("status") == "error":
                    print(f"Error: {res.get('error')}")
                elif res.get("status") == "assignment":
                    print(f"=> {res['variable']} = {res['formatted']}")
                elif res.get("status") == "equation_solved":
                    sol = res['solution']
                    print("Solution:")
                    print(json.dumps(sol, indent=2, default=str))
                else:
                    print(f"=> {res.get('formatted')}")

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def run_test_suite() -> int:
    """Discover and run all unittests in the package."""
    print("=" * 60)
    print(" Running ScientificCalculator Automated Test Suite...")
    print("=" * 60)
    loader = unittest.TestLoader()
    here = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(os.path.join(here, "tests"), top_level_dir=here)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1

def main():
    parser = argparse.ArgumentParser(description="Scientific Calculator in Python")
    parser.add_argument("-e", "--eval", type=str, help="Evaluate a single mathematical expression or equation")
    parser.add_argument("--mode", type=str, choices=["RAD", "DEG", "GRAD"], default="RAD", help="Trigonometric angle mode")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("-t", "--test", action="store_true", help="Run automated test suite")
    
    args = parser.parse_args()

    if args.test:
        sys.exit(run_test_suite())

    cli = ScientificCalculatorCLI(angle_mode=args.mode)

    if args.eval:
        sys.exit(cli.run_eval(args.eval, output_json=args.json))
    else:
        cli.run_repl()

if __name__ == "__main__":
    main()
