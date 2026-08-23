"""
CLI and integration tests for ScientificCalculator.
"""
import unittest
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from calculator import ScientificCalculatorCLI, run_test_suite
from commands import handle_command
from engine import CalculatorEngine

class TestCalculatorCLI(unittest.TestCase):
    def setUp(self):
        self.cli = ScientificCalculatorCLI(angle_mode="RAD")
        self.engine = CalculatorEngine(angle_mode="RAD")

    def test_cli_eval_simple(self):
        f = io.StringIO()
        with redirect_stdout(f):
            ret = self.cli.run_eval("2 + 3 * 4")
        self.assertEqual(ret, 0)
        self.assertIn("14", f.getvalue().strip())

    def test_cli_eval_deg_trig(self):
        cli_deg = ScientificCalculatorCLI(angle_mode="DEG")
        f = io.StringIO()
        with redirect_stdout(f):
            ret = cli_deg.run_eval("sin(90)")
        self.assertEqual(ret, 0)
        self.assertIn("1", f.getvalue().strip())

    def test_cli_eval_json(self):
        f = io.StringIO()
        with redirect_stdout(f):
            ret = self.cli.run_eval("sqrt(16)", output_json=True)
        self.assertEqual(ret, 0)
        self.assertIn('"formatted": "4"', f.getvalue())

    def test_cli_eval_error_handling(self):
        f_err = io.StringIO()
        with redirect_stderr(f_err):
            ret = self.cli.run_eval("10 / 0")
        self.assertEqual(ret, 1)
        self.assertIn("Division by zero", f_err.getvalue())

    def test_command_help(self):
        handled, msg = handle_command(":help", self.engine)
        self.assertTrue(handled)
        self.assertIn("ScientificCalculator", msg)

    def test_command_mode_switch(self):
        handled, msg = handle_command(":mode DEG", self.engine)
        self.assertTrue(handled)
        self.assertEqual(self.engine.get_angle_mode(), "DEG")

    def test_command_constants(self):
        handled, msg = handle_command(":consts", self.engine)
        self.assertTrue(handled)
        self.assertIn("pi", msg)
        self.assertIn("c", msg)

    def test_command_functions_list(self):
        handled, msg = handle_command(":funcs", self.engine)
        self.assertTrue(handled)
        self.assertIn("sin", msg)
        self.assertIn("sqrt", msg)

if __name__ == "__main__":
    unittest.main()
