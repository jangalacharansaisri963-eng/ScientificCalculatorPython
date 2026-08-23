"""
complex_math.py
A comprehensive library for complex number arithmetic, polar coordinates,
fixed values, transformations, and identities built strictly in pure Python.
"""


# =====================================================================
# 0. NATIVE MATH HELPERS CLASS (Replaces math and cmath)
# =====================================================================


class MathHelpers:
    """Pure Python numeric algorithms using Taylor series and Newton-Raphson."""

    PI = 3.14159265358979323846264338327950288
    E = 2.71828182845904523536028747135266249

    @staticmethod
    def abs(x: float) -> float:
        return x if x >= 0 else -x

    @staticmethod
    def factorial(n: int) -> int:
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    @staticmethod
    def sqrt(x: float) -> float:
        if x < 0:
            raise ValueError("Use complex_sqrt for negative numbers.")
        if x == 0:
            return 0.0
        guess = x / 2.0
        for _ in range(50):
            guess = 0.5 * (guess + x / guess)
        return guess

    @staticmethod
    def exp(x: float) -> float:
        """Taylor series expansion for e^x."""
        res = 1.0
        term = 1.0
        for i in range(1, 40):
            term *= x / i
            res += term
        return res

    @staticmethod
    def ln(x: float) -> float:
        """Natural logarithm using Newton-Raphson."""
        if x <= 0:
            raise ValueError("Domain error for real logarithm.")
        y = 0.0 if x < 2 else x / MathHelpers.E
        for _ in range(50):
            ey = MathHelpers.exp(y)
            y = y + 2 * (x - ey) / (x + ey)
        return y

    @staticmethod
    def sin(x: float) -> float:
        """Taylor series for sin(x)."""
        x = x % (2 * MathHelpers.PI)
        res = 0.0
        term = x
        for i in range(1, 30, 2):
            res += term
            term *= -x * x / ((i + 1) * (i + 2))
        return res

    @staticmethod
    def cos(x: float) -> float:
        """Taylor series for cos(x)."""
        x = x % (2 * MathHelpers.PI)
        res = 0.0
        term = 1.0
        for i in range(0, 30, 2):
            res += term
            term *= -x * x / ((i + 1) * (i + 2))
        return res

    @staticmethod
    def atan2(y: float, x: float) -> float:
        """Four-quadrant inverse tangent via Taylor series & reductions."""
        if x == 0:
            if y > 0:
                return MathHelpers.PI / 2
            if y < 0:
                return -MathHelpers.PI / 2
            return 0.0

        z = y / x
        if MathHelpers.abs(z) < 1.0:
            # Taylor series for |z| < 1
            atan = 0.0
            term = z
            for i in range(1, 60, 2):
                atan += term
                term *= -z * z * i / (i + 2)
        else:
            # Identity atan(z) = pi/2 - atan(1/z) for |z| >= 1
            inv_z = 1.0 / z
            atan = 0.0
            term = inv_z
            for i in range(1, 60, 2):
                atan += term
                term *= -inv_z * inv_z * i / (i + 2)
            atan = (
                (MathHelpers.PI / 2) - atan
                if z > 0
                else (-MathHelpers.PI / 2) - atan
            )

        if x < 0:
            return atan + MathHelpers.PI if y >= 0 else atan - MathHelpers.PI
        return atan


# =====================================================================
# 1. FIXED VALUES & IMAGINARY CONSTANTS
# =====================================================================

I = complex(0.0, 1.0)
NEG_I = complex(0.0, -1.0)
ZERO = complex(0.0, 0.0)
ONE = complex(1.0, 0.0)
NEG_ONE = complex(-1.0, 0.0)
EULER_I = complex(MathHelpers.E, 0.0)
PI_COMPLEX = complex(MathHelpers.PI, 0.0)
HALF_I = complex(0.0, 0.5)


# =====================================================================
# 2. CONSTRUCTORS & BASIC ACCESSORS
# =====================================================================


def make_complex(real: float, imag: float = 0.0) -> complex:
    return complex(real, imag)


def real_part(z: complex) -> float:
    return z.real


def imag_part(z: complex) -> float:
    return z.imag


def is_pure_real(z: complex, tol: float = 1e-12) -> bool:
    return MathHelpers.abs(z.imag) < tol


def is_pure_imaginary(z: complex, tol: float = 1e-12) -> bool:
    return MathHelpers.abs(z.real) < tol


def is_zero_complex(z: complex, tol: float = 1e-12) -> bool:
    return complex_magnitude(z) < tol


def complex_conjugate(z: complex) -> complex:
    return complex(z.real, -z.imag)


def complex_magnitude(z: complex) -> float:
    return MathHelpers.sqrt(z.real**2 + z.imag**2)


def complex_magnitude_sq(z: complex) -> float:
    return z.real**2 + z.imag**2


def complex_phase_rad(z: complex) -> float:
    return MathHelpers.atan2(z.imag, z.real)


def complex_phase_deg(z: complex) -> float:
    return complex_phase_rad(z) * (180.0 / MathHelpers.PI)


def rad_to_deg(rad: float) -> float:
    return rad * (180.0 / MathHelpers.PI)


def deg_to_rad(deg: float) -> float:
    return deg * (MathHelpers.PI / 180.0)


# =====================================================================
# 3. POLAR & RECTANGULAR CONVERSIONS
# =====================================================================


def polar_to_cartesian(r: float, theta: float, mode: str = "RAD") -> complex:
    rad = deg_to_rad(theta) if mode.upper() == "DEG" else theta
    return complex(r * MathHelpers.cos(rad), r * MathHelpers.sin(rad))


def cartesian_to_polar(z: complex, mode: str = "RAD") -> tuple[float, float]:
    r = complex_magnitude(z)
    theta = complex_phase_rad(z)
    if mode.upper() == "DEG":
        theta = rad_to_deg(theta)
    return r, theta


def polar_multiply(
    r1: float, theta1: float, r2: float, theta2: float
) -> tuple[float, float]:
    return r1 * r2, theta1 + theta2


def polar_divide(
    r1: float, theta1: float, r2: float, theta2: float
) -> tuple[float, float]:
    return r1 / r2, theta1 - theta2


# =====================================================================
# 4. ARITHMETIC OPERATIONS
# =====================================================================


def complex_add(z1: complex, z2: complex) -> complex:
    return complex(z1.real + z2.real, z1.imag + z2.imag)


def complex_subtract(z1: complex, z2: complex) -> complex:
    return complex(z1.real - z2.real, z1.imag - z2.imag)


def complex_multiply(z1: complex, z2: complex) -> complex:
    return complex(
        z1.real * z2.real - z1.imag * z2.imag,
        z1.real * z2.imag + z1.imag * z2.real,
    )


def complex_divide(z1: complex, z2: complex) -> complex:
    denom = z2.real**2 + z2.imag**2
    if denom == 0:
        raise ZeroDivisionError("Complex division by zero.")
    return complex(
        (z1.real * z2.real + z1.imag * z2.imag) / denom,
        (z1.imag * z2.real - z1.real * z2.imag) / denom,
    )


def complex_scale(z: complex, scalar: float) -> complex:
    return complex(z.real * scalar, z.imag * scalar)


def complex_negate(z: complex) -> complex:
    return complex(-z.real, -z.imag)


def complex_inverse(z: complex) -> complex:
    return complex_divide(ONE, z)


def imag_multiply(z: complex, k: float) -> complex:
    """Multiplies z by k*i."""
    return complex_multiply(z, complex(0, k))


def imag_divide(z: complex, k: float) -> complex:
    """Divides z by k*i."""
    return complex_divide(z, complex(0, k))


# =====================================================================
# 5. EXPONENTIATION, LOGARITHMS, & ROOTS
# =====================================================================


def complex_exp(z: complex) -> complex:
    """Computes e^z = e^x * (cos(y) + i*sin(y))."""
    ex = MathHelpers.exp(z.real)
    return complex(ex * MathHelpers.cos(z.imag), ex * MathHelpers.sin(z.imag))


def complex_log(z: complex) -> complex:
    """Principal branch natural logarithm of z."""
    r = complex_magnitude(z)
    theta = complex_phase_rad(z)
    return complex(MathHelpers.ln(r), theta)


def complex_log_base(z: complex, base: complex) -> complex:
    return complex_divide(complex_log(z), complex_log(base))


def complex_power(z: complex, w: complex) -> complex:
    """Calculates z^w = e^(w * ln(z))."""
    if z == 0:
        return ZERO
    return complex_exp(complex_multiply(w, complex_log(z)))


def complex_power_real(z: complex, n: float) -> complex:
    r, theta = cartesian_to_polar(z)
    return polar_to_cartesian(r**n, theta * n)


def complex_sqrt(z: complex) -> complex:
    r = complex_magnitude(z)
    u = MathHelpers.sqrt((r + z.real) / 2.0)
    v = MathHelpers.sqrt((r - z.real) / 2.0)
    if z.imag < 0:
        v = -v
    return complex(u, v)


def complex_nth_roots(z: complex, n: int) -> list[complex]:
    """Computes all n roots of z using De Moivre's Theorem."""
    r, theta = cartesian_to_polar(z)
    r_root = r ** (1.0 / n)
    roots = []
    for k in range(n):
        angle = (theta + 2 * MathHelpers.PI * k) / n
        roots.append(polar_to_cartesian(r_root, angle))
    return roots


# =====================================================================
# 6. TRIGONOMETRIC & HYPERBOLIC FUNCTIONS
# =====================================================================


def complex_sin(z: complex) -> complex:
    # sin(x + iy) = sin(x)cosh(y) + i*cos(x)sinh(y)
    x, y = z.real, z.imag
    ey, eny = MathHelpers.exp(y), MathHelpers.exp(-y)
    cosh_y = (ey + eny) / 2.0
    sinh_y = (ey - eny) / 2.0
    return complex(
        MathHelpers.sin(x) * cosh_y, MathHelpers.cos(x) * sinh_y
    )


def complex_cos(z: complex) -> complex:
    # cos(x + iy) = cos(x)cosh(y) - i*sin(x)sinh(y)
    x, y = z.real, z.imag
    ey, eny = MathHelpers.exp(y), MathHelpers.exp(-y)
    cosh_y = (ey + eny) / 2.0
    sinh_y = (ey - eny) / 2.0
    return complex(
        MathHelpers.cos(x) * cosh_y, -MathHelpers.sin(x) * sinh_y
    )


def complex_tan(z: complex) -> complex:
    return complex_divide(complex_sin(z), complex_cos(z))


def complex_sec(z: complex) -> complex:
    return complex_divide(ONE, complex_cos(z))


def complex_csc(z: complex) -> complex:
    return complex_divide(ONE, complex_sin(z))


def complex_cot(z: complex) -> complex:
    return complex_divide(complex_cos(z), complex_sin(z))


def complex_sinh(z: complex) -> complex:
    return complex_scale(
        complex_subtract(complex_exp(z), complex_exp(complex_negate(z))), 0.5
    )


def complex_cosh(z: complex) -> complex:
    return complex_scale(
        complex_add(complex_exp(z), complex_exp(complex_negate(z))), 0.5
    )


def complex_tanh(z: complex) -> complex:
    return complex_divide(complex_sinh(z), complex_cosh(z))


# =====================================================================
# 7. VECTOR, DISTANCE & GEOMETRIC TRANSFORMATIONS
# =====================================================================


def complex_distance(z1: complex, z2: complex) -> float:
    return complex_magnitude(complex_subtract(z1, z2))


def complex_midpoint(z1: complex, z2: complex) -> complex:
    return complex_scale(complex_add(z1, z2), 0.5)


def complex_dot_product(z1: complex, z2: complex) -> float:
    """Vector dot product: Re(z1 * conj(z2))."""
    return z1.real * z2.real + z1.imag * z2.imag


def complex_cross_product(z1: complex, z2: complex) -> float:
    """2D vector cross product: Im(conj(z1) * z2)."""
    return z1.real * z2.imag - z1.imag * z2.real


def rotate_complex(z: complex, angle_rad: float) -> complex:
    """Rotates z by an angle in radians around origin."""
    rotator = polar_to_cartesian(1.0, angle_rad)
    return complex_multiply(z, rotator)


def rotate_around_point(
    z: complex, center: complex, angle_rad: float
) -> complex:
    """Rotates z around an arbitrary complex center point."""
    shifted = complex_subtract(z, center)
    rotated = rotate_complex(shifted, angle_rad)
    return complex_add(rotated, center)


def complex_translate(z: complex, dz: complex) -> complex:
    return complex_add(z, dz)


def complex_reflect_real_axis(z: complex) -> complex:
    return complex_conjugate(z)


def complex_reflect_imag_axis(z: complex) -> complex:
    return complex(-z.real, z.imag)


def complex_reflect_origin(z: complex) -> complex:
    return complex_negate(z)


def mobius_transform(
    z: complex, a: complex, b: complex, c: complex, d: complex
) -> complex:
    """Calculates f(z) = (az + b) / (cz + d)."""
    num = complex_add(complex_multiply(a, z), b)
    den = complex_add(complex_multiply(c, z), d)
    return complex_divide(num, den)


def joukowsky_transform(z: complex) -> complex:
    """Aerodynamic map: J(z) = z + 1/z."""
    return complex_add(z, complex_inverse(z))


# =====================================================================
# 8. SIGNAL PROCESSING & ENGINEERING UTILITIES
# =====================================================================


def parallel_impedance(z1: complex, z2: complex) -> complex:
    """Parallel electrical impedance: (z1 * z2) / (z1 + z2)."""
    return complex_divide(
        complex_multiply(z1, z2), complex_add(z1, z2)
    )


def series_impedance(z1: complex, z2: complex) -> complex:
    return complex_add(z1, z2)


def dft_single_bin(signal: list[float], k: int) -> complex:
    """Calculates discrete Fourier transform bin k."""
    N = len(signal)
    acc = ZERO
    for n in range(N):
        angle = -2 * MathHelpers.PI * k * n / N
        twiddle = polar_to_cartesian(1.0, angle)
        acc = complex_add(acc, complex_scale(twiddle, signal[n]))
    return acc


# =====================================================================
# 9. TWENTY FUNDAMENTAL COMPLEX IDENTITIES & VERIFICATIONS
# =====================================================================


def identity_euler_formula(theta: float, tol: float = 1e-9) -> bool:
    """1. e^(i*theta) == cos(theta) + i*sin(theta)"""
    lhs = complex_exp(complex_multiply(I, complex(theta, 0)))
    rhs = complex(MathHelpers.cos(theta), MathHelpers.sin(theta))
    return complex_distance(lhs, rhs) < tol


def identity_euler_identity(tol: float = 1e-9) -> bool:
    """2. e^(i*pi) + 1 == 0"""
    lhs = complex_add(complex_exp(complex_multiply(I, PI_COMPLEX)), ONE)
    return complex_magnitude(lhs) < tol


def identity_i_squared(tol: float = 1e-9) -> bool:
    """3. i^2 == -1"""
    return complex_distance(complex_multiply(I, I), NEG_ONE) < tol


def identity_i_to_i(tol: float = 1e-9) -> bool:
    """4. i^i == e^(-pi/2)"""
    lhs = complex_power(I, I)
    rhs = complex(MathHelpers.exp(-MathHelpers.PI / 2), 0)
    return complex_distance(lhs, rhs) < tol


def identity_modulus_squared(z: complex, tol: float = 1e-9) -> bool:
    """5. |z|^2 == z * conj(z)"""
    lhs = complex_magnitude_sq(z)
    rhs = complex_multiply(z, complex_conjugate(z)).real
    return MathHelpers.abs(lhs - rhs) < tol


def identity_triangle_inequality(
    z1: complex, z2: complex, tol: float = 1e-9
) -> bool:
    """6. |z1 + z2| <= |z1| + |z2|"""
    lhs = complex_magnitude(complex_add(z1, z2))
    rhs = complex_magnitude(z1) + complex_magnitude(z2)
    return lhs <= rhs + tol


def identity_de_moivre(r: float, theta: float, n: int, tol: float = 1e-9) -> bool:
    """7. (r*(cos t + i sin t))^n == r^n * (cos(n*t) + i sin(n*t))"""
    z = polar_to_cartesian(r, theta)
    lhs = complex_power_real(z, n)
    rhs = polar_to_cartesian(r**n, theta * n)
    return complex_distance(lhs, rhs) < tol


def identity_sin_squared_cos_squared(z: complex, tol: float = 1e-9) -> bool:
    """8. sin^2(z) + cos^2(z) == 1"""
    s = complex_sin(z)
    c = complex_cos(z)
    sum_sq = complex_add(complex_multiply(s, s), complex_multiply(c, c))
    return complex_distance(sum_sq, ONE) < tol


def identity_cosh_squared_sinh_squared(z: complex, tol: float = 1e-9) -> bool:
    """9. cosh^2(z) - sinh^2(z) == 1"""
    ch = complex_cosh(z)
    sh = complex_sinh(z)
    diff_sq = complex_subtract(complex_multiply(ch, ch), complex_multiply(sh, sh))
    return complex_distance(diff_sq, ONE) < tol


def identity_sin_hyperbolic_relation(z: complex, tol: float = 1e-9) -> bool:
    """10. sin(i*z) == i*sinh(z)"""
    lhs = complex_sin(complex_multiply(I, z))
    rhs = complex_multiply(I, complex_sinh(z))
    return complex_distance(lhs, rhs) < tol


def identity_cos_hyperbolic_relation(z: complex, tol: float = 1e-9) -> bool:
    """11. cos(i*z) == cosh(z)"""
    lhs = complex_cos(complex_multiply(I, z))
    rhs = complex_cosh(z)
    return complex_distance(lhs, rhs) < tol


def identity_log_product(z1: complex, z2: complex, tol: float = 1e-9) -> bool:
    """12. ln(z1 * z2) == ln(z1) + ln(z2) (mod 2*pi*i)"""
    lhs = complex_log(complex_multiply(z1, z2))
    rhs = complex_add(complex_log(z1), complex_log(z2))
    diff = complex_subtract(lhs, rhs)
    return MathHelpers.abs(diff.imag % (2 * MathHelpers.PI)) < tol


def identity_conjugate_product(
    z1: complex, z2: complex, tol: float = 1e-9
) -> bool:
    """13. conj(z1 * z2) == conj(z1) * conj(z2)"""
    lhs = complex_conjugate(complex_multiply(z1, z2))
    rhs = complex_multiply(complex_conjugate(z1), complex_conjugate(z2))
    return complex_distance(lhs, rhs) < tol


def identity_conjugate_quotient(
    z1: complex, z2: complex, tol: float = 1e-9
) -> bool:
    """14. conj(z1 / z2) == conj(z1) / conj(z2)"""
    lhs = complex_conjugate(complex_divide(z1, z2))
    rhs = complex_divide(complex_conjugate(z1), complex_conjugate(z2))
    return complex_distance(lhs, rhs) < tol


def identity_sqrt_i(tol: float = 1e-9) -> bool:
    """15. sqrt(i) == (1 + i) / sqrt(2)"""
    lhs = complex_sqrt(I)
    inv_sqrt2 = 1.0 / MathHelpers.sqrt(2)
    rhs = complex(inv_sqrt2, inv_sqrt2)
    return complex_distance(lhs, rhs) < tol


def identity_log_i(tol: float = 1e-9) -> bool:
    """16. ln(i) == i * pi / 2"""
    lhs = complex_log(I)
    rhs = complex(0, MathHelpers.PI / 2)
    return complex_distance(lhs, rhs) < tol


def identity_sin_euler(z: complex, tol: float = 1e-9) -> bool:
    """17. sin(z) == (e^(iz) - e^(-iz)) / (2i)"""
    iz = complex_multiply(I, z)
    neg_iz = complex_negate(iz)
    num = complex_subtract(complex_exp(iz), complex_exp(neg_iz))
    lhs = complex_divide(num, complex(0, 2))
    rhs = complex_sin(z)
    return complex_distance(lhs, rhs) < tol


def identity_cos_euler(z: complex, tol: float = 1e-9) -> bool:
    """18. cos(z) == (e^(iz) + e^(-iz)) / 2"""
    iz = complex_multiply(I, z)
    neg_iz = complex_negate(iz)
    num = complex_add(complex_exp(iz), complex_exp(neg_iz))
    lhs = complex_scale(num, 0.5)
    rhs = complex_cos(z)
    return complex_distance(lhs, rhs) < tol


def identity_inverse_property(z: complex, tol: float = 1e-9) -> bool:
    """19. z * (1/z) == 1"""
    lhs = complex_multiply(z, complex_inverse(z))
    return complex_distance(lhs, ONE) < tol


def identity_rotation_four_times(z: complex, tol: float = 1e-9) -> bool:
    """20. Rotating z four times by pi/2 returns z (z * i^4 == z)"""
    res = z
    for _ in range(4):
        res = complex_multiply(res, I)
    return complex_distance(res, z) < tol
