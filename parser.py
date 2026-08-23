"""
Mathematical expression parser and token evaluator using Shunting-yard algorithm.
Supports unary operators, implicit multiplication, function calls with arbitrary arguments, and constants.
"""
from typing import Any, Callable
import re
import math
from constants import CONSTANTS_DICT
from functions import get_function
from functions import basic_math
from functions import factorial as fact_mod

class TokenType:
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    FUNCTION = "FUNCTION"
    OPERATOR = "OPERATOR"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    FACTORIAL = "FACTORIAL"

class Token:
    def __init__(self, type_: str, value: Any, arg_count: int = 1):
        self.type = type_
        self.value = value
        self.arg_count = arg_count

    def __repr__(self):
        if self.type == TokenType.FUNCTION:
            return f"Token(FUNCTION, {self.value}, args={self.arg_count})"
        return f"Token({self.type}, {self.value})"

# Operator precedence and associativity
OPERATORS = {
    '+': {'prec': 2, 'assoc': 'L', 'binary': True},
    '-': {'prec': 2, 'assoc': 'L', 'binary': True},
    '*': {'prec': 3, 'assoc': 'L', 'binary': True},
    '/': {'prec': 3, 'assoc': 'L', 'binary': True},
    '%': {'prec': 3, 'assoc': 'L', 'binary': True},
    'NEG': {'prec': 4, 'assoc': 'R', 'binary': False},  # Unary minus
    'POS': {'prec': 4, 'assoc': 'R', 'binary': False},  # Unary plus
    '^': {'prec': 5, 'assoc': 'R', 'binary': True},
    '**': {'prec': 5, 'assoc': 'R', 'binary': True},
}

def tokenize(expression: str) -> list[Token]:
    """Tokenize a mathematical expression string into a list of Tokens."""
    raw_tokens: list[Token] = []
    i = 0
    s = expression.strip()
    n = len(s)
    
    while i < n:
        char = s[i]
        
        # Whitespace
        if char.isspace():
            i += 1
            continue
            
        # Number: digits, decimal point, or complex imaginary suffix 'j' / 'i'
        if char.isdigit() or (char == '.' and i + 1 < n and s[i + 1].isdigit()):
            start = i
            has_dot = (char == '.')
            while i < n and (s[i].isdigit() or (s[i] == '.' and not has_dot)):
                if s[i] == '.':
                    has_dot = True
                i += 1
            # Check scientific notation (e.g. 1e-5 or 2.5e3)
            if i < n and s[i].lower() == 'e' and (i + 1 < n) and (s[i + 1].isdigit() or s[i + 1] in '+-'):
                i += 1
                if i < n and s[i] in '+-':
                    i += 1
                while i < n and s[i].isdigit():
                    i += 1
                    
            num_str = s[start:i]
            # Check if followed by imaginary unit 'j' or 'i' (e.g. 3.5j)
            if i < n and s[i].lower() in ('j', 'i'):
                i += 1
                val = complex(0, float(num_str))
                raw_tokens.append(Token(TokenType.NUMBER, val))
            else:
                val = float(num_str) if '.' in num_str or 'e' in num_str.lower() else int(num_str)
                raw_tokens.append(Token(TokenType.NUMBER, val))
            continue
            
        # Identifier (function, constant, variable)
        if char.isalpha() or char == '_':
            start = i
            while i < n and (s[i].isalnum() or s[i] == '_'):
                i += 1
            name = s[start:i]
            # Check imaginary unit 'j' or 'i' standalone
            if name in ('j', 'i') and (not raw_tokens or raw_tokens[-1].type != TokenType.IDENTIFIER):
                raw_tokens.append(Token(TokenType.NUMBER, complex(0, 1)))
            else:
                raw_tokens.append(Token(TokenType.IDENTIFIER, name))
            continue
            
        # Factorial operator
        if char == '!':
            raw_tokens.append(Token(TokenType.FACTORIAL, '!'))
            i += 1
            continue
            
        # Parentheses and commas
        if char == '(':
            raw_tokens.append(Token(TokenType.LPAREN, '('))
            i += 1
            continue
        if char == ')':
            raw_tokens.append(Token(TokenType.RPAREN, ')'))
            i += 1
            continue
        if char == ',':
            raw_tokens.append(Token(TokenType.COMMA, ','))
            i += 1
            continue
            
        # Power '**'
        if char == '*' and i + 1 < n and s[i + 1] == '*':
            raw_tokens.append(Token(TokenType.OPERATOR, '^'))
            i += 2
            continue
            
        # Standard operators
        if char in "+-*/%^":
            raw_tokens.append(Token(TokenType.OPERATOR, char))
            i += 1
            continue
            
        raise ValueError(f"Unexpected character in expression: '{char}' at index {i}")
        
    # Distinguish FUNCTION vs IDENTIFIER (variable/constant)
    classified_tokens: list[Token] = []
    for idx, t in enumerate(raw_tokens):
        if t.type == TokenType.IDENTIFIER and idx + 1 < len(raw_tokens) and raw_tokens[idx + 1].type == TokenType.LPAREN:
            classified_tokens.append(Token(TokenType.FUNCTION, t.value))
        else:
            classified_tokens.append(t)

    # Insert implicit multiplications (e.g. 2pi -> 2 * pi, 3(4) -> 3 * (4), (2)(3) -> (2)*(3))
    expanded_tokens: list[Token] = []
    for idx, t in enumerate(classified_tokens):
        expanded_tokens.append(t)
        if idx + 1 < len(classified_tokens):
            next_t = classified_tokens[idx + 1]
            if (t.type == TokenType.NUMBER and next_t.type in (TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.LPAREN)) or \
               (t.type == TokenType.IDENTIFIER and next_t.type in (TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.LPAREN)) or \
               (t.type == TokenType.RPAREN and next_t.type in (TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.LPAREN, TokenType.NUMBER)) or \
               (t.type == TokenType.FACTORIAL and next_t.type in (TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.NUMBER, TokenType.LPAREN)):
                expanded_tokens.append(Token(TokenType.OPERATOR, '*'))
                
    # Disambiguate unary plus/minus
    processed_tokens: list[Token] = []
    for idx, t in enumerate(expanded_tokens):
        if t.type == TokenType.OPERATOR and t.value in ('+', '-'):
            # It is unary if it is the first token, or preceded by an operator, LPAREN, or COMMA
            if idx == 0 or expanded_tokens[idx - 1].type in (TokenType.OPERATOR, TokenType.LPAREN, TokenType.COMMA):
                unary_op = 'POS' if t.value == '+' else 'NEG'
                processed_tokens.append(Token(TokenType.OPERATOR, unary_op))
                continue
        processed_tokens.append(t)
        
    return processed_tokens

def parse_to_rpn(tokens: list[Token]) -> list[Any]:
    """Shunting-yard algorithm to convert infix tokens into Reverse Polish Notation (RPN)."""
    output_queue: list[Any] = []
    operator_stack: list[Any] = []
    
    # Track argument counts for function calls
    # When LPAREN is pushed, if preceded by FUNCTION, we track its arg count
    arg_counts: list[int] = []
    has_args: list[bool] = []
    
    for i, token in enumerate(tokens):
        if token.type == TokenType.NUMBER:
            output_queue.append(token)
            if has_args:
                has_args[-1] = True
            
        elif token.type == TokenType.IDENTIFIER:
            # Standalone variable or constant
            output_queue.append(token)
            if has_args:
                has_args[-1] = True
            
        elif token.type == TokenType.FUNCTION:
            operator_stack.append(token)
            
        elif token.type == TokenType.FACTORIAL:
            output_queue.append(token)
            
        elif token.type == TokenType.COMMA:
            while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                output_queue.append(operator_stack.pop())
            if not operator_stack:
                raise ValueError("Mismatched comma or parentheses")
            if arg_counts:
                arg_counts[-1] += 1
                
        elif token.type == TokenType.OPERATOR:
            o1 = token.value
            while operator_stack:
                top = operator_stack[-1]
                if top.type == TokenType.OPERATOR:
                    o2 = top.value
                    prec1 = OPERATORS[o1]['prec']
                    prec2 = OPERATORS[o2]['prec']
                    assoc1 = OPERATORS[o1]['assoc']
                    if (assoc1 == 'L' and prec1 <= prec2) or (assoc1 == 'R' and prec1 < prec2):
                        output_queue.append(operator_stack.pop())
                    else:
                        break
                elif top.type == TokenType.FUNCTION:
                    output_queue.append(operator_stack.pop())
                else:
                    break
            operator_stack.append(token)
            
        elif token.type == TokenType.LPAREN:
            operator_stack.append(token)
            # Check if this LPAREN belongs to a function call
            if len(operator_stack) >= 2 and operator_stack[-2].type == TokenType.FUNCTION:
                # Next token might be RPAREN (0 arguments)
                is_empty = (i + 1 < len(tokens) and tokens[i + 1].type == TokenType.RPAREN)
                arg_counts.append(0 if is_empty else 1)
                has_args.append(not is_empty)
            else:
                arg_counts.append(0)
                has_args.append(False)
            
        elif token.type == TokenType.RPAREN:
            while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                output_queue.append(operator_stack.pop())
            if not operator_stack:
                raise ValueError("Mismatched parentheses (extra closing parenthesis)")
            operator_stack.pop()  # Pop LPAREN
            
            argc = arg_counts.pop() if arg_counts else 0
            if has_args:
                has_args.pop()
                
            # If top of operator stack is a FUNCTION, pop it and set its argument count
            if operator_stack and operator_stack[-1].type == TokenType.FUNCTION:
                fn_token = operator_stack.pop()
                fn_token.arg_count = argc
                output_queue.append(fn_token)
                
    while operator_stack:
        top = operator_stack.pop()
        if top.type in (TokenType.LPAREN, TokenType.RPAREN):
            raise ValueError("Mismatched parentheses (unclosed parenthesis)")
        output_queue.append(top)
        
    return output_queue

def evaluate_rpn(rpn: list[Any],
                 variables: dict[str, Any] = None,
                 angle_mode: str = "RAD",
                 custom_funcs: dict[str, Callable] = None) -> Any:
    """Evaluate RPN token list."""
    if variables is None:
        variables = {}
    if custom_funcs is None:
        custom_funcs = {}
        
    stack: list[Any] = []
    
    for item in rpn:
        if isinstance(item, Token):
            if item.type == TokenType.NUMBER:
                stack.append(item.value)
                
            elif item.type == TokenType.FACTORIAL:
                if not stack:
                    raise ValueError("Syntax error: factorial missing operand")
                val = stack.pop()
                stack.append(fact_mod.factorial_func(val))
                
            elif item.type == TokenType.IDENTIFIER:
                name = item.value
                # Check variable
                if name in variables:
                    stack.append(variables[name])
                # Check constant
                elif name in CONSTANTS_DICT:
                    stack.append(CONSTANTS_DICT[name])
                else:
                    raise ValueError(f"Unknown identifier: '{name}'")

            elif item.type == TokenType.FUNCTION:
                name = item.value
                argc = item.arg_count
                func = custom_funcs.get(name) or get_function(name)
                
                if func is None:
                    raise ValueError(f"Unknown function: '{name}'")
                    
                if len(stack) < argc:
                    raise ValueError(f"Function '{name}' requires {argc} arguments, but only {len(stack)} provided")
                    
                args = [stack.pop() for _ in range(argc)]
                args.reverse()
                
                # Check if function supports angle mode
                import inspect
                sig = inspect.signature(func)
                if 'mode' in sig.parameters:
                    res = func(*args, mode=angle_mode)
                else:
                    res = func(*args)
                stack.append(res)
                    
            elif item.type == TokenType.OPERATOR:
                op = item.value
                if op == 'NEG':
                    if not stack:
                        raise ValueError("Syntax error: unary minus missing operand")
                    stack.append(-stack.pop())
                elif op == 'POS':
                    if not stack:
                        raise ValueError("Syntax error: unary plus missing operand")
                    # No-op
                elif op == '+':
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.add(a, b))
                elif op == '-':
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.subtract(a, b))
                elif op == '*':
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.multiply(a, b))
                elif op == '/':
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.divide(a, b))
                elif op == '%':
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.modulo(a, b))
                elif op in ('^', '**'):
                    b, a = stack.pop(), stack.pop()
                    stack.append(basic_math.power(a, b))
                else:
                    raise ValueError(f"Unsupported operator: {op}")
        else:
            stack.append(item)
            
    if len(stack) != 1:
        raise ValueError(f"Malformed expression: multiple values left on stack ({len(stack)})")
        
    return stack[0]

def parse_and_eval(expression: str,
                   variables: dict[str, Any] = None,
                   angle_mode: str = "RAD",
                   custom_funcs: dict[str, Callable] = None) -> Any:
    """Convenience helper to tokenize, parse to RPN, and evaluate an expression in one step."""
    tokens = tokenize(expression)
    rpn = parse_to_rpn(tokens)
    return evaluate_rpn(rpn, variables=variables, angle_mode=angle_mode, custom_funcs=custom_funcs)

