"""
polynomials.py
A standard library for polynomial algebra, calculus, and numerical operations.
Built strictly using pure Python (No `import math` or `import cmath`).
"""

# =====================================================================
# 0. NATIVE MATH HELPERS CLASS
# =====================================================================


class MathHelpers:
    """Pure-Python implementations replacing 'math' and 'cmath' modules."""

    @staticmethod
    def abs(x):
        """Absolute value for real and complex numbers."""
        if isinstance(x, complex):
            return (x.real**2 + x.imag**2) ** 0.5
        return x if x >= 0 else -x

    @staticmethod
    def factorial(n):
        """Calculates n! using pure loops."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers.")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def sqrt(x):
        """Calculates square roots for positive, negative, or complex numbers

        using Newton's method.
        """
        if isinstance(x, complex):
            a, b = x.real, x.imag
            mod = (a**2 + b**2) ** 0.5
            real_part = ((mod + a) / 2) ** 0.5
            imag_part = ((mod - a) / 2) ** 0.5
            if b < 0:
                imag_part = -imag_part
            return complex(real_part, imag_part)

        if x < 0:
            return complex(0, (-x) ** 0.5)

        if x == 0:
            return 0.0

        guess = x / 2.0
        for _ in range(50):
            guess = 0.5 * (guess + x / guess)
        return guess


# =====================================================================
# 1. CORE POLYNOMIAL CLASS
# =====================================================================


class Polynomial:
    """Core Polynomial class handling general poly operations of any degree."""

    def __init__(self, coefficients):
        """Coefficients ordered from lowest degree to highest:

        c0 + c1*x + c2*x^2 ...
        """
        coeffs = list(coefficients)
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        self.coeffs = coeffs if coeffs else [0]

    @property
    def degree(self):
        return len(self.coeffs) - 1

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{i}" if c != 1 else f"x^{i}")
        return " + ".join(terms) if terms else "0"

    def __call__(self, x):
        """Evaluate polynomial at x using Horner's method."""
        result = 0
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __add__(self, other):
        other = other if isinstance(other, Polynomial) else Polynomial([other])
        max_len = max(len(self.coeffs), len(other.coeffs))
        c1 = self.coeffs + [0] * (max_len - len(self.coeffs))
        c2 = other.coeffs + [0] * (max_len - len(other.coeffs))
        return Polynomial([a + b for a, b in zip(c1, c2)])

    def __sub__(self, other):
        other = other if isinstance(other, Polynomial) else Polynomial([other])
        max_len = max(len(self.coeffs), len(other.coeffs))
        c1 = self.coeffs + [0] * (max_len - len(self.coeffs))
        c2 = other.coeffs + [0] * (max_len - len(other.coeffs))
        return Polynomial([a - b for a, b in zip(c1, c2)])

    def __mul__(self, other):
        other = other if isinstance(other, Polynomial) else Polynomial([other])
        res = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                res[i + j] += c1 * c2
        return Polynomial(res)

    def derivative(self):
        if self.degree == 0:
            return Polynomial([0])
        return Polynomial([i * c for i, c in enumerate(self.coeffs[1:], start=1)])

    def integrate(self, C=0):
        res = [C] + [c / (i + 1) for i, c in enumerate(self.coeffs)]
        return Polynomial(res)


# =====================================================================
# 2. SCHOOL ALGEBRA FUNCTIONS
# =====================================================================


def constant_poly(c):
    return Polynomial([c])


def linear_poly(a, b):
    """a*x + b"""
    return Polynomial([b, a])


def quadratic_poly(a, b, c):
    """a*x^2 + b*x + c"""
    return Polynomial([c, b, a])


def cubic_poly(a, b, c, d):
    """a*x^3 + b*x^2 + c*x + d"""
    return Polynomial([d, c, b, a])


def quadratic_roots(a, b, c):
    """Solves ax^2 + bx + c = 0 via quadratic formula."""
    disc = b**2 - 4 * a * c
    sqrt_disc = MathHelpers.sqrt(disc)
    r1 = (-b + sqrt_disc) / (2 * a)
    r2 = (-b - sqrt_disc) / (2 * a)
    return r1, r2


def quadratic_vertex(a, b, c):
    """Returns (h, k) vertex coordinates of a parabola."""
    h = -b / (2 * a)
    k = c - (b**2) / (4 * a)
    return h, k


def quadratic_discriminant(a, b, c):
    return b**2 - 4 * a * c


def evaluate_poly(coeffs, x):
    """Evaluates polynomial given list of coeffs [c0, c1, ...] at x."""
    return Polynomial(coeffs)(x)


def add_polynomials(p1_coeffs, p2_coeffs):
    return (Polynomial(p1_coeffs) + Polynomial(p2_coeffs)).coeffs


def subtract_polynomials(p1_coeffs, p2_coeffs):
    return (Polynomial(p1_coeffs) - Polynomial(p2_coeffs)).coeffs


def multiply_polynomials(p1_coeffs, p2_coeffs):
    return (Polynomial(p1_coeffs) * Polynomial(p2_coeffs)).coeffs


def poly_degree(coeffs):
    return Polynomial(coeffs).degree


def is_monic(coeffs):
    p = Polynomial(coeffs)
    return p.coeffs[-1] == 1


def make_monic(coeffs):
    p = Polynomial(coeffs)
    lead = p.coeffs[-1]
    return [c / lead for c in p.coeffs]


def synthetic_division(coeffs, r):
    """Divides p(x) by (x - r). Returns (quotient_coeffs, remainder)."""
    p = Polynomial(coeffs)
    rev_c = list(reversed(p.coeffs))
    quotient = [rev_c[0]]
    for c in rev_c[1:-1]:
        quotient.append(c + quotient[-1] * r)
    remainder = rev_c[-1] + quotient[-1] * r
    return list(reversed(quotient)), remainder


def polynomial_long_division(num_coeffs, den_coeffs):
    """Divides two polynomials. Returns (quotient, remainder)."""
    n = list(Polynomial(num_coeffs).coeffs)
    d = list(Polynomial(den_coeffs).coeffs)
    if len(n) < len(d):
        return [0], n
    out = [0] * (len(n) - len(d) + 1)
    for i in range(len(out) - 1, -1, -1):
        coeff = n[i + len(d) - 1] / d[-1]
        out[i] = coeff
        for j in range(len(d)):
            n[i + j] -= coeff * d[j]
    return Polynomial(out).coeffs, Polynomial(n).coeffs


def expand_roots(roots):
    """Constructs polynomial coeffs from a list of roots."""
    p = Polynomial([1])
    for r in roots:
        p = p * Polynomial([-r, 1])
    return p.coeffs


def vieta_sum_of_roots(coeffs):
    """Returns -a_{n-1} / a_n."""
    p = Polynomial(coeffs)
    return -p.coeffs[-2] / p.coeffs[-1]


def vieta_product_of_roots(coeffs):
    """Returns (-1)^n * a_0 / a_n."""
    p = Polynomial(coeffs)
    n = p.degree
    return ((-1) ** n) * p.coeffs[0] / p.coeffs[-1]


# =====================================================================
# 3. COLLEGE CALCULUS & ANALYSIS FUNCTIONS
# =====================================================================


def poly_derivative(coeffs):
    return Polynomial(coeffs).derivative().coeffs


def poly_nth_derivative(coeffs, n):
    p = Polynomial(coeffs)
    for _ in range(n):
        p = p.derivative()
    return p.coeffs


def poly_indefinite_integral(coeffs, C=0):
    return Polynomial(coeffs).integrate(C).coeffs


def poly_definite_integral(coeffs, a, b):
    P = Polynomial(coeffs).integrate()
    return P(b) - P(a)


def find_critical_points(coeffs):
    """Finds real critical points using derivative roots."""
    p_prime = Polynomial(coeffs).derivative()
    if p_prime.degree == 1:
        return [-p_prime.coeffs[0] / p_prime.coeffs[1]]
    elif p_prime.degree == 2:
        a, b, c = p_prime.coeffs[2], p_prime.coeffs[1], p_prime.coeffs[0]
        r1, r2 = quadratic_roots(a, b, c)
        results = []
        for r in (r1, r2):
            if isinstance(r, complex):
                if MathHelpers.abs(r.imag) < 1e-9:
                    results.append(r.real)
            else:
                results.append(r)
        return results
    return []


def taylor_sine(n_terms):
    """Taylor series polynomial for sin(x) around 0 up to n terms."""
    coeffs = [0] * (2 * n_terms)
    for k in range(n_terms):
        coeffs[2 * k + 1] = ((-1) ** k) / MathHelpers.factorial(2 * k + 1)
    return Polynomial(coeffs).coeffs


def taylor_cosine(n_terms):
    """Taylor series polynomial for cos(x) around 0 up to n terms."""
    coeffs = [0] * (2 * n_terms)
    for k in range(n_terms):
        coeffs[2 * k] = ((-1) ** k) / MathHelpers.factorial(2 * k)
    return Polynomial(coeffs).coeffs


def taylor_exp(n_terms):
    """Taylor series polynomial for e^x around 0 up to n terms."""
    coeffs = [1 / MathHelpers.factorial(k) for k in range(n_terms)]
    return Polynomial(coeffs).coeffs


def newton_raphson_root(coeffs, initial_guess, max_iter=100, tol=1e-7):
    """Finds a single root using Newton-Raphson method."""
    p = Polynomial(coeffs)
    dp = p.derivative()
    x = initial_guess
    for _ in range(max_iter):
        y = p(x)
        if MathHelpers.abs(y) < tol:
            return x
        dy = dp(x)
        if dy == 0:
            break
        x = x - y / dy
    return x


def lagrange_interpolation(x_points, y_points):
    """Generates polynomial interpolating given (x,y) points."""
    total_poly = Polynomial([0])
    n = len(x_points)
    for i in range(n):
        term = Polynomial([1])
        for j in range(n):
            if i != j:
                factor = Polynomial(
                    [
                        -x_points[j] / (x_points[i] - x_points[j]),
                        1 / (x_points[i] - x_points[j]),
                    ]
                )
                term = term * factor
        total_poly = total_poly + (term * y_points[i])
    return total_poly.coeffs


# =====================================================================
# 4. SPECIAL & ORTHOGONAL POLYNOMIALS (Advanced College/Physics)
# =====================================================================


def legendre_poly(n):
    """Generates n-th Legendre Polynomial P_n(x)."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    p0 = Polynomial([1])
    p1 = Polynomial([0, 1])
    for i in range(1, n):
        term1 = Polynomial([0, (2 * i + 1) / (i + 1)]) * p1
        term2 = p0 * (i / (i + 1))
        p_next = term1 - term2
        p0, p1 = p1, p_next
    return p1.coeffs


def chebyshev_t(n):
    """Generates Chebyshev polynomial of first kind T_n(x)."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 1]
    t0 = Polynomial([1])
    t1 = Polynomial([0, 1])
    for _ in range(2, n + 1):
        t_next = (Polynomial([0, 2]) * t1) - t0
        t0, t1 = t1, t_next
    return t1.coeffs


def hermite_poly(n):
    """Generates n-th Physicist's Hermite Polynomial H_n(x)."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    h0 = Polynomial([1])
    h1 = Polynomial([0, 2])
    for i in range(1, n):
        h_next = (Polynomial([0, 2]) * h1) - (h0 * (2 * i))
        h0, h1 = h1, h_next
    return h1.coeffs


def laguerre_poly(n):
    """Generates n-th Laguerre Polynomial L_n(x)."""
    if n == 0:
        return [1]
    if n == 1:
        return [1, -1]
    l0 = Polynomial([1])
    l1 = Polynomial([1, -1])
    for i in range(1, n):
        term1 = Polynomial([2 * i + 1, -1]) * l1
        term2 = l0 * (i**2)
        l_next = (term1 - term2) * (1 / (i + 1))
        l0, l1 = l1, l_next
    return l1.coeffs
