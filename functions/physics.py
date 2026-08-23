"""
Physics formulas, scientific calculators, and precision geometry utilities.
"""

from decimal import Decimal, getcontext
from fractions import Fraction
import math
from typing import Any, Tuple, Union

# Set precision for Decimal operations
getcontext().prec = 28


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Physical Constants (Decimal Precision)
PLANCK_CONSTANT = Decimal('6.62607015e-34')
PI_DEC = Decimal(str(math.pi))

# Physical Constants (Float Precision)
C_LIGHT = 299792458.0  # Speed of light in m/s
G_GRAVITATIONAL = 6.67430e-11  # Gravitational constant
H_PLANCK = 6.62607015e-34  # Planck constant
HBAR = 1.054571817e-34  # Reduced Planck constant
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant
E_CHARGE = 1.602176634e-19  # Elementary charge
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity
MU_0 = 1.25663706212e-6  # Vacuum permeability
R_GAS = 8.314462618  # Universal gas constant
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant
G_EARTH = 9.80665  # Standard gravity on Earth


# ==============================================================================
# 0. BASIC MECHANICS & PARTICLES (PRECISION ALIGNED)
# ==============================================================================

def pressure(force: Any, area: Any) -> Decimal:
    return Decimal(str(force)) / Decimal(str(area))


def boltzman_entropy(microstates: int) -> Decimal:
    """Calculates exact Boltzmann entropy using Decimal for precision.
    
    Formula: S = k_B * ln(Ω)
    """
    if microstates <= 0:
        raise ValueError("Microstates must be greater than zero.")
        
    k_B = Decimal("1.380649e-23")
    omega = Decimal(str(microstates))
    return k_B * omega.ln()


def radians(degrees: Union[int, float]) -> str:
    deg = int(degrees)
    frac = Fraction(deg, 180)
    
    num = frac.numerator
    den = frac.denominator
    
    if num == 0:
        return "0"
        
    num_part = "" if num == 1 else ("-" if num == -1 else str(num))
    den_part = "" if den == 1 else f"/{den}"
    
    return f"{num_part}π{den_part}"


def degrees(rad_str: str) -> Union[float, int]:
    s = str(rad_str).strip().replace(" ", "").replace("pi", "π")
    
    if "π" not in s:
        return float(s) * (180.0 / math.pi)
        
    if s == "π":
        return 180
    if s == "-π":
        return -180
        
    parts = s.split("π")
    coefficient_part = parts[0]
    denominator_part = parts[1] if len(parts) > 1 else ""
    
    if coefficient_part == "" or coefficient_part == "+":
        num = 1
    elif coefficient_part == "-":
        num = -1
    else:
        num = int(coefficient_part)
        
    if denominator_part.startswith("/"):
        den = int(denominator_part[1:])
    else:
        den = 1
        
    result = (Fraction(num, den)) * 180
    return float(result) if result.denominator != 1 else int(result)


def time_from_velocity(displacement: Any, velocity: Any) -> Decimal:
    return Decimal(str(displacement)) / Decimal(str(velocity))


def time_from_work(work: Any, force: Any) -> Decimal:
    return Decimal(str(work)) / Decimal(str(force))


def time_from_power(work: Any, power: Any) -> Decimal:
    return Decimal(str(work)) / Decimal(str(power))
    

def gravitational_constant(force: Any, distance: Any, mass1: Any, mass2: Any) -> Decimal:
    f_dec = Decimal(str(force))
    d_dec = Decimal(str(distance))
    m1_dec = Decimal(str(mass1))
    m2_dec = Decimal(str(mass2))
    
    denominator = m1_dec * m2_dec
    if denominator == Decimal('0'):
        raise ZeroDivisionError("Masses cannot be zero.")
        
    return (f_dec * (d_dec ** 2)) / denominator
    
                
def momentum(mass: Any, velocity: Any) -> Decimal:
    return Decimal(str(mass)) * Decimal(str(velocity))

                
def gravitational_potential(force: Any, displacement: Any, mass: Any) -> Decimal:
    work_val = Decimal(str(force)) * Decimal(str(displacement))
    return work_val / Decimal(str(mass))


def displacement_hypotenuse(a: Any, b: Any) -> Decimal:
    sum_of_squares = Decimal(str(a))**2 + Decimal(str(b))**2
    return sum_of_squares.sqrt()


def right_angled_adjacent(opposite: Any, hypotenuse: Any) -> Decimal:
    diff_of_squares = Decimal(str(hypotenuse))**2 - Decimal(str(opposite))**2
    if diff_of_squares < 0:
        raise ValueError("Invalid dimensions: opposite side cannot be >= hypotenuse for a right triangle.")
    return diff_of_squares.sqrt()


def direction_of_displacement(dx: Any, dy: Any) -> Decimal:
    dec_dx = float(Decimal(str(dx)))
    dec_dy = float(Decimal(str(dy)))
    angle_deg = math.degrees(math.atan2(dec_dy, dec_dx))
    return Decimal(str(angle_deg))


def impulse(force: Any, time: Any) -> Decimal:
    return Decimal(str(force)) * Decimal(str(time))


def latent_heat(heat_energy_joules: Any, mass_kg: Any) -> Decimal:
    return Decimal(str(heat_energy_joules)) / Decimal(str(mass_kg))


def calorific_value(total_heat_joules: Any, mass_kg: Any) -> Decimal:
    return Decimal(str(total_heat_joules)) / Decimal(str(mass_kg))


def heat_for_phase_change(mass_kg: Any, latent_heat_constant: Any) -> Decimal:
    return Decimal(str(mass_kg)) * Decimal(str(latent_heat_constant))


def heat_from_combustion(mass_fuel_kg: Any, calorific_value_constant: Any) -> Decimal:
    return Decimal(str(mass_fuel_kg)) * Decimal(str(calorific_value_constant))


def specific_heat_capacity(heat: Any, mass: Any, delta_temperature: Any) -> Decimal:
    mass_term = Decimal(str(mass)) * Decimal(str(delta_temperature))
    return Decimal(str(heat)) / mass_term


def heat_for_temperature_change(mass: Any, specific_heat_constant: Any, delta_temperature: Any) -> Decimal:
    return (Decimal(str(mass)) * Decimal(str(specific_heat_constant)) * Decimal(str(delta_temperature)))


def area(force: Any, pressure: Any) -> Decimal:
    return Decimal(str(force)) / Decimal(str(pressure))


def volume(length: Any) -> Decimal:
    side = Decimal(str(length))
    return side ** 3


def energy(h: Any, frequency: Any) -> Decimal:
    """E = h * f. Uses provided h or falls back to PLANCK_CONSTANT."""
    h_dec = Decimal(str(h)) if h is not None else PLANCK_CONSTANT
    return h_dec * Decimal(str(frequency))


def frequency(energy: Any, h: Any) -> Decimal:
    """f = E / h. Uses provided h or falls back to PLANCK_CONSTANT."""
    h_dec = Decimal(str(h)) if h is not None else PLANCK_CONSTANT
    return Decimal(str(energy)) / h_dec


def diagonal_square(side: Any) -> Decimal:
    return Decimal(str(side)) * Decimal('2').sqrt()


def diagonal_rectangle(length: Any, width: Any) -> Decimal:
    l_dec = Decimal(str(length))
    w_dec = Decimal(str(width))
    return (l_dec**2 + w_dec**2).sqrt()


def side(area: Any) -> Decimal:
    return Decimal(str(area)).sqrt()


def weight(mass: Any, g: Any) -> Decimal:
    """Calculates weight (W = m * g)."""
    return Decimal(str(mass)) * Decimal(str(g))


# ==============================================================================
# 1. GEOMETRIC KINEMATICS
# ==============================================================================

def distance_circle(revolution: Any, radius: Any = 1) -> Decimal:
    """Calculates total path distance around a circle."""
    rev = Decimal(str(revolution))
    rad = Decimal(str(radius))
    return rev * Decimal('2') * PI_DEC * rad


def displacement_circle(revolution: Any, radius: Any = 1) -> Decimal:
    """Calculates straight-line displacement chord using precise Decimals."""
    rev = Decimal(str(revolution))
    rad = Decimal(str(radius))

    fractional_rev = (rev % Decimal('1'))

    if fractional_rev == Decimal('0'):
        return Decimal('0')
    if fractional_rev == Decimal('0.5'):
        return Decimal('2') * rad
    if fractional_rev in [Decimal('0.25'), Decimal('0.75')]:
        return Decimal('2').sqrt() * rad

    angle = float(fractional_rev * Decimal('2') * PI_DEC)
    sin_value = Decimal(str(math.sin(angle / 2.0)))
    return Decimal('2') * rad * abs(sin_value)


def chord_length(radius: Any, angle_degrees: Any) -> Decimal:
    """Calculates chord length using formula: 2 * R * sin(theta / 2)."""
    r = Decimal(str(radius))
    theta = Decimal(str(angle_degrees))
    half_angle_rad = float(theta / Decimal('2') * PI_DEC / Decimal('180'))
    sin_val = Decimal(str(math.sin(half_angle_rad)))
    return Decimal('2') * r * sin_val


# Backward-compatible alias for chord_length_from_angle
chord_length_from_angle = chord_length


# ==============================================================================
# 2. CORE KINEMATICS & DYNAMICS
# ==============================================================================

def calculate_velocity(distance: Any, time: Any) -> Decimal:
    t = Decimal(str(time))
    if t == Decimal('0'):
        raise ZeroDivisionError("Time cannot be zero.")
    return Decimal(str(distance)) / t


def calculate_acceleration(initial_velocity: Any, final_velocity: Any, time: Any) -> Decimal:
    t = Decimal(str(time))
    if t == Decimal('0'):
        raise ZeroDivisionError("Time cannot be zero.")
    return (Decimal(str(final_velocity)) - Decimal(str(initial_velocity))) / t


def calculate_force(mass: Any, acceleration: Any) -> Decimal:
    return Decimal(str(mass)) * Decimal(str(acceleration))


# ==============================================================================
# 3. FRICTION ENGINE
# ==============================================================================

def calculate_friction(mu: Any, mass: Any, gravity: Any = "9.80665") -> Decimal:
    coefficient = Decimal(str(mu))
    m = Decimal(str(mass))
    g = Decimal(str(gravity))

    if coefficient < 0:
        raise ValueError("Friction coefficient (mu) cannot be negative.")
    if m < 0:
        raise ValueError("Mass cannot be negative.")
    return coefficient * m * g


# ==============================================================================
# 4. RESULTANT FLUID DENSITIES
# ==============================================================================

def resultant_density_by_volume(density1: Any, density2: Any) -> Decimal:
    return (Decimal(str(density1)) + Decimal(str(density2))) / Decimal('2')


def resultant_density_by_mass(density1: Any, density2: Any) -> Decimal:
    rho1 = Decimal(str(density1))
    rho2 = Decimal(str(density2))
    denominator = rho1 + rho2
    if denominator == Decimal('0'):
        raise ZeroDivisionError("Sum of densities cannot be zero.")
    return (Decimal('2') * rho1 * rho2) / denominator


def resultant_density_general(mass1: Any, volume1: Any, mass2: Any, volume2: Any) -> Decimal:
    total_volume = Decimal(str(volume1)) + Decimal(str(volume2))
    if total_volume == Decimal('0'):
        raise ZeroDivisionError("Total volume cannot be zero.")
    return (Decimal(str(mass1)) + Decimal(str(mass2))) / total_volume


# ==============================================================================
# 5. UNIFIED TEMPERATURE CONVERSION ENGINE
# ==============================================================================

def convert_temperature(value: Any, from_unit: str, to_unit: str) -> Decimal:
    val = Decimal(str(value))
    src = from_unit.strip().lower()
    dst = to_unit.strip().lower()

    if src in ['c', 'celsius', 'degree', 'degrees']:
        celsius = val
    elif src in ['f', 'fahrenheit']:
        celsius = (val - Decimal('32')) * Decimal('5') / Decimal('9')
    elif src in ['k', 'kelvin']:
        celsius = val - Decimal('273.15')
    elif src in ['r', 'reamur', 'réaumur']:
        celsius = val * Decimal('5') / Decimal('4')
    else:
        raise ValueError(f"Unknown source temperature unit: '{from_unit}'")

    if dst in ['c', 'celsius', 'degree', 'degrees']:
        return celsius
    elif dst in ['f', 'fahrenheit']:
        return (celsius * Decimal('9') / Decimal('5')) + Decimal('32')
    elif dst in ['k', 'kelvin']:
        result = celsius + Decimal('273.15')
        if result < Decimal('0'):
            print("[Warning]: Result is below Absolute Zero (0K)!")
        return result
    elif dst in ['r', 'reamur', 'réaumur']:
        return celsius * Decimal('4') / Decimal('5')
    else:
        raise ValueError(f"Unknown target temperature unit: '{to_unit}'")


# ==============================================================================
# 6. MIGRATED FUNCTIONS FROM BASIC_MATH
# ==============================================================================

def distance(speed: Any, time: Any) -> Decimal:
    return Decimal(str(speed)) * Decimal(str(time))


def speed(distance: Any, time: Any) -> Decimal:
    t = Decimal(str(time))
    if t == Decimal('0'):
        raise ZeroDivisionError("Time cannot be zero.")
    return Decimal(str(distance)) / t


def time(distance: Any, speed: Any) -> Decimal:
    s = Decimal(str(speed))
    if s == Decimal('0'):
        raise ZeroDivisionError("Speed cannot be zero.")
    return Decimal(str(distance)) / s


def work(force: Any, distance: Any) -> Decimal:
    return Decimal(str(force)) * Decimal(str(distance))


def power(work: Any, time: Any) -> Decimal:
    t = Decimal(str(time))
    if t == Decimal('0'):
        raise ZeroDivisionError("Time cannot be zero.")
    return Decimal(str(work)) / t


def kineticenergy(mass: Any, velocity: Any) -> Decimal:
    m = Decimal(str(mass))
    v = Decimal(str(velocity))
    return Decimal('0.5') * m * (v ** 2)


def potentialenergy(mass: Any, gravity: Any, height: Any) -> Decimal:
    m = Decimal(str(mass))
    g = Decimal(str(gravity))
    h = Decimal(str(height))
    return m * g * h


def density(mass: Any, volume: Any) -> Decimal:
    v = Decimal(str(volume))
    if v == Decimal('0'):
        raise ZeroDivisionError("Volume cannot be zero.")
    return Decimal(str(mass)) / v


def mass(density: Any, volume: Any) -> Decimal:
    return Decimal(str(density)) * Decimal(str(volume))


def volume_by_density(mass: Any, density: Any) -> Decimal:
    d = Decimal(str(density))
    if d == Decimal('0'):
        raise ZeroDivisionError("Density cannot be zero.")
    return Decimal(str(mass)) / d

                                  
# ==============================================================================
# 7. PERIMETERS & AREAS
# ==============================================================================

def circle_circumference(radius: Any) -> Decimal:
    r = Decimal(str(radius))
    return Decimal('2') * PI_DEC * r

def circle_area(radius: Any) -> Decimal:
    r = Decimal(str(radius))
    return PI_DEC * (r ** 2)

def semicircle_area(radius: Any) -> Decimal:
    return circle_area(radius) / Decimal('2')

def semicircle_perimeter(radius: Any) -> Decimal:
    r = Decimal(str(radius))
    return PI_DEC * r + Decimal('2') * r

def ellipse_area(a: Any, b: Any) -> Decimal:
    return PI_DEC * Decimal(str(a)) * Decimal(str(b))

def ellipse_perimeter(a: Any, b: Any) -> Decimal:
    a_d = Decimal(str(a))
    b_d = Decimal(str(b))
    term = (Decimal('3') * (a_d + b_d) - ((Decimal('3') * a_d + b_d) * (a_d + Decimal('3') * b_d)).sqrt())
    return PI_DEC * term

def square_perimeter(side: Any) -> Decimal:
    s = Decimal(str(side))
    return Decimal('4') * s

def square_area(side: Any) -> Decimal:
    s = Decimal(str(side))
    return s ** 2

def rectangle_perimeter(length: Any, width: Any) -> Decimal:
    l = Decimal(str(length))
    w = Decimal(str(width))
    return Decimal('2') * (l + w)

def rectangle_area(length: Any, width: Any) -> Decimal:
    return Decimal(str(length)) * Decimal(str(width))

def parallelogram_perimeter(base: Any, side: Any) -> Decimal:
    b = Decimal(str(base))
    s = Decimal(str(side))
    return Decimal('2') * (b + s)

def parallelogram_area(base: Any, height: Any) -> Decimal:
    return Decimal(str(base)) * Decimal(str(height))

def triangle_perimeter(a: Any, b: Any, c: Any) -> Decimal:
    return Decimal(str(a)) + Decimal(str(b)) + Decimal(str(c))

def triangle_area_heron(a: Any, b: Any, c: Any) -> Decimal:
    a_d = Decimal(str(a))
    b_d = Decimal(str(b))
    c_d = Decimal(str(c))
    s = (a_d + b_d + c_d) / Decimal('2')
    inner = s * (s - a_d) * (s - b_d) * (s - c_d)
    if inner < 0:
        raise ValueError("Invalid triangle sides for area calculation (negative square root).")
    return inner.sqrt()

def triangle_area_base_height(base: Any, height: Any) -> Decimal:
    return Decimal(str(base)) * Decimal(str(height)) / Decimal('2')

def right_triangle_hypotenuse(a: Any, b: Any) -> Decimal:
    a_d = Decimal(str(a))
    b_d = Decimal(str(b))
    return (a_d**2 + b_d**2).sqrt()

def right_triangle_area(a: Any, b: Any) -> Decimal:
    return Decimal(str(a)) * Decimal(str(b)) / Decimal('2')

def equilateral_triangle_area(side: Any) -> Decimal:
    s = Decimal(str(side))
    sqrt_3 = Decimal('3').sqrt()
    return (sqrt_3 / Decimal('4')) * (s ** 2)

def equilateral_triangle_perimeter(side: Any) -> Decimal:
    return Decimal('3') * Decimal(str(side))

def isosceles_triangle_area(base: Any, equal_side: Any) -> Decimal:
    b = Decimal(str(base))
    s = Decimal(str(equal_side))
    inner = s**2 - (b**2 / Decimal('4'))
    if inner < 0:
        raise ValueError("Invalid dimensions for isosceles triangle (imaginary height).")
    h = inner.sqrt()
    return (b * h) / Decimal('2')

def trapezoid_area(a: Any, b: Any, height: Any) -> Decimal:
    return (Decimal(str(a)) + Decimal(str(b))) * Decimal(str(height)) / Decimal('2')

def trapezoid_perimeter(a: Any, b: Any, c: Any, d: Any) -> Decimal:
    return Decimal(str(a)) + Decimal(str(b)) + Decimal(str(c)) + Decimal(str(d))

def rhombus_area_by_diagonals(d1: Any, d2: Any) -> Decimal:
    return (Decimal(str(d1)) * Decimal(str(d2))) / Decimal('2')

def rhombus_perimeter_from_diagonals(d1: Any, d2: Any) -> Decimal:
    half1 = (Decimal(str(d1)) / Decimal('2'))
    half2 = (Decimal(str(d2)) / Decimal('2'))
    side = (half1**2 + half2**2).sqrt()
    return Decimal('4') * side

def kite_area_by_diagonals(d1: Any, d2: Any) -> Decimal:
    return (Decimal(str(d1)) * Decimal(str(d2))) / Decimal('2')

def kite_perimeter(side1: Any, side2: Any) -> Decimal:
    return Decimal('2') * (Decimal(str(side1)) + Decimal(str(side2)))

def regular_polygon_perimeter(n_sides: Any, side_length: Any) -> Decimal:
    n = Decimal(str(n_sides))
    s = Decimal(str(side_length))
    return n * s

def regular_polygon_area(n_sides: Any, side_length: Any) -> Decimal:
    n = int(n_sides)
    s = float(side_length)
    if n < 3:
        raise ValueError("A polygon must have at least 3 sides.")
    apothem = s / (2.0 * math.tan(math.pi / n))
    area_val = 0.5 * n * s * apothem
    return Decimal(str(area_val))

def regular_pentagon_area(side: Any) -> Decimal:
    return regular_polygon_area(5, side)

def regular_pentagon_perimeter(side: Any) -> Decimal:
    return regular_polygon_perimeter(5, side)

def regular_hexagon_area(side: Any) -> Decimal:
    s = Decimal(str(side))
    sqrt_3 = Decimal('3').sqrt()
    return (Decimal('3') * sqrt_3 / Decimal('2')) * (s ** 2)

def regular_hexagon_perimeter(side: Any) -> Decimal:
    return Decimal('6') * Decimal(str(side))

def regular_octagon_area(side: Any) -> Decimal:
    s = Decimal(str(side))
    sqrt_2 = Decimal('2').sqrt()
    return Decimal('2') * (Decimal('1') + sqrt_2) * (s ** 2)

def regular_octagon_perimeter(side: Any) -> Decimal:
    return Decimal('8') * Decimal(str(side))

def annulus_area(R: Any, r: Any) -> Decimal:
    R_d = Decimal(str(R))
    r_d = Decimal(str(r))
    return PI_DEC * (R_d**2 - r_d**2)

def annulus_perimeter(R: Any, r: Any) -> Decimal:
    R_d = Decimal(str(R))
    r_d = Decimal(str(r))
    return Decimal('2') * PI_DEC * (R_d + r_d)

def sector_area(radius: Any, angle_degrees: Any) -> Decimal:
    r = Decimal(str(radius))
    theta = Decimal(str(angle_degrees)) * PI_DEC / Decimal('180')
    return (r ** 2) * theta / Decimal('2')

def sector_arc_length(radius: Any, angle_degrees: Any) -> Decimal:
    r = Decimal(str(radius))
    theta = Decimal(str(angle_degrees)) * PI_DEC / Decimal('180')
    return r * theta


# ==============================================================================
# 8. VOLUMES
# ==============================================================================

def cube_volume(side: Any) -> Decimal:
    s = Decimal(str(side))
    return s ** 3

def cuboid_volume(length: Any, width: Any, height: Any) -> Decimal:
    return Decimal(str(length)) * Decimal(str(width)) * Decimal(str(height))

def sphere_volume(radius: Any) -> Decimal:
    r = Decimal(str(radius))
    return (Decimal('4') / Decimal('3')) * PI_DEC * (r ** 3)

def hemisphere_volume(radius: Any) -> Decimal:
    r = Decimal(str(radius))
    return (Decimal('2') / Decimal('3')) * PI_DEC * (r ** 3)

def cylinder_volume(radius: Any, height: Any) -> Decimal:
    r = Decimal(str(radius))
    h = Decimal(str(height))
    return PI_DEC * (r ** 2) * h

def cone_volume(radius: Any, height: Any) -> Decimal:
    r = Decimal(str(radius))
    h = Decimal(str(height))
    return (Decimal('1') / Decimal('3')) * PI_DEC * (r ** 2) * h

def frustum_cone_volume(r1: Any, r2: Any, height: Any) -> Decimal:
    R1 = Decimal(str(r1))
    R2 = Decimal(str(r2))
    h = Decimal(str(height))
    return (Decimal('1') / Decimal('3')) * PI_DEC * h * (R1**2 + R1*R2 + R2**2)

def pyramid_volume(base_area: Any, height: Any) -> Decimal:
    return Decimal(str(base_area)) * Decimal(str(height)) / Decimal('3')

def square_pyramid_volume(side: Any, height: Any) -> Decimal:
    base = Decimal(str(side)) ** 2
    return base * Decimal(str(height)) / Decimal('3')

def rectangular_pyramid_volume(length: Any, width: Any, height: Any) -> Decimal:
    base = Decimal(str(length)) * Decimal(str(width))
    return base * Decimal(str(height)) / Decimal('3')

def prism_volume(base_area: Any, length: Any) -> Decimal:
    return Decimal(str(base_area)) * Decimal(str(length))

def triangular_prism_volume(base: Any, height_of_triangle: Any, length: Any) -> Decimal:
    base_area = Decimal(str(base)) * Decimal(str(height_of_triangle)) / Decimal('2')
    return base_area * Decimal(str(length))

def regular_tetrahedron_volume(side: Any) -> Decimal:
    s = Decimal(str(side))
    sqrt_2 = Decimal('2').sqrt()
    denom = Decimal('6') * sqrt_2
    return s**3 / denom

def ellipsoid_volume(a: Any, b: Any, c: Any) -> Decimal:
    return (Decimal('4') / Decimal('3')) * PI_DEC * Decimal(str(a)) * Decimal(str(b)) * Decimal(str(c))

def torus_volume(R: Any, r: Any) -> Decimal:
    R_d = Decimal(str(R))
    r_d = Decimal(str(r))
    return Decimal('2') * (PI_DEC ** 2) * R_d * (r_d ** 2)

def hollow_cylinder_volume(R_outer: Any, R_inner: Any, height: Any) -> Decimal:
    Ro = Decimal(str(R_outer))
    Ri = Decimal(str(R_inner))
    h = Decimal(str(height))
    if Ri >= Ro:
        raise ValueError("Inner radius must be smaller than outer radius.")
    return PI_DEC * h * (Ro**2 - Ri**2)


# ==============================================================================
# 9. ADVANCED GEOMETRY & TRIGONOMETRIC IDENTITIES
# ==============================================================================

def sagitta(radius: Any, chord_len: Any) -> Decimal:
    """Calculates sagitta (arc height): h = R - sqrt(R^2 - (c/2)^2)."""
    r = Decimal(str(radius))
    c = Decimal(str(chord_len))
    half_c = c / Decimal('2')
    if r < half_c:
        raise ValueError("Radius cannot be smaller than half of the chord length.")
    return r - (r**2 - half_c**2).sqrt()


def circular_segment_area(radius: Any, angle_degrees: Any) -> Decimal:
    """Area of a circular segment: 0.5 * R^2 * (theta_rad - sin(theta))."""
    r = Decimal(str(radius))
    theta_deg = Decimal(str(angle_degrees))
    theta_rad = float(theta_deg * PI_DEC / Decimal('180'))
    term = Decimal(str(theta_rad)) - Decimal(str(math.sin(theta_rad)))
    return Decimal('0.5') * (r ** 2) * term


def apothem_regular_polygon(n_sides: Any, side_length: Any) -> Decimal:
    """Apothem of a regular polygon: s / (2 * tan(pi / n))."""
    n = int(n_sides)
    s = float(side_length)
    if n < 3:
        raise ValueError("A polygon must have at least 3 sides.")
    apothem = s / (2.0 * math.tan(math.pi / n))
    return Decimal(str(apothem))


def sphere_surface_area(radius: Any) -> Decimal:
    """Surface area of a sphere: 4 * pi * R^2."""
    r = Decimal(str(radius))
    return Decimal('4') * PI_DEC * (r ** 2)


def hemisphere_total_surface_area(radius: Any) -> Decimal:
    """Total surface area of a solid hemisphere: 3 * pi * R^2."""
    r = Decimal(str(radius))
    return Decimal('3') * PI_DEC * (r ** 2)


def cylinder_total_surface_area(radius: Any, height: Any) -> Decimal:
    """Total surface area of a cylinder: 2 * pi * R * (R + H)."""
    r = Decimal(str(radius))
    h = Decimal(str(height))
    return Decimal('2') * PI_DEC * r * (r + h)


def cone_slant_height(radius: Any, height: Any) -> Decimal:
    """Slant height of a cone: sqrt(R^2 + H^2)."""
    r = Decimal(str(radius))
    h = Decimal(str(height))
    return (r**2 + h**2).sqrt()


def cone_total_surface_area(radius: Any, height: Any) -> Decimal:
    """Total surface area of a cone: pi * R * (R + slant_height)."""
    r = Decimal(str(radius))
    l = cone_slant_height(radius, height)
    return PI_DEC * r * (r + l)


# ==============================================================================
# 10. ADVANCED PHYSICS & WAVE MECHANICS
# ==============================================================================

def wave_speed(frequency_hz: Any, wavelength_m: Any) -> Decimal:
    """Wave speed: v = f * lambda."""
    return Decimal(str(frequency_hz)) * Decimal(str(wavelength_m))


def wave_frequency(speed_ms: Any, wavelength_m: Any) -> Decimal:
    """Frequency from wave speed: f = v / lambda."""
    lam = Decimal(str(wavelength_m))
    if lam == Decimal('0'):
        raise ZeroDivisionError("Wavelength cannot be zero.")
    return Decimal(str(speed_ms)) / lam


def wavelength(speed_ms: Any, frequency_hz: Any) -> Decimal:
    """Wavelength from wave speed: lambda = v / f."""
    f = Decimal(str(frequency_hz))
    if f == Decimal('0'):
        raise ZeroDivisionError("Frequency cannot be zero.")
    return Decimal(str(speed_ms)) / f


def kinetic_energy_relativistic(mass_val: Any, velocity: Any, speed_of_light: Any = "299792458") -> Decimal:
    """Relativistic kinetic energy: KE = (gamma - 1) * m * c^2."""
    m = Decimal(str(mass_val))
    v = Decimal(str(velocity))
    c = Decimal(str(speed_of_light))
    if v >= c:
        raise ValueError("Velocity cannot exceed or equal the speed of light.")
    gamma = Decimal('1') / (Decimal('1') - (v / c)**2).sqrt()
    return (gamma - Decimal('1')) * m * (c ** 2)


def centripetal_force(mass_val: Any, velocity: Any, radius: Any) -> Decimal:
    """Centripetal force: Fc = (m * v^2) / R."""
    m = Decimal(str(mass_val))
    v = Decimal(str(velocity))
    r = Decimal(str(radius))
    if r == Decimal('0'):
        raise ZeroDivisionError("Radius cannot be zero.")
    return (m * (v ** 2)) / r


def centripetal_acceleration(velocity: Any, radius: Any) -> Decimal:
    """Centripetal acceleration: ac = v^2 / R."""
    v = Decimal(str(velocity))
    r = Decimal(str(radius))
    if r == Decimal('0'):
        raise ZeroDivisionError("Radius cannot be zero.")
    return (v ** 2) / r


def gravitational_force(mass1: Any, mass2: Any, distance_val: Any, G: Any = "6.67430e-11") -> Decimal:
    """Newton's Law of Gravitation: F = G * (m1 * m2) / r^2."""
    g_dec = Decimal(str(G))
    m1 = Decimal(str(mass1))
    m2 = Decimal(str(mass2))
    r = Decimal(str(distance_val))
    if r == Decimal('0'):
        raise ZeroDivisionError("Distance cannot be zero.")
    return g_dec * (m1 * m2) / (r ** 2)


def electric_force(q1: Any, q2: Any, distance_val: Any, k: Any = "8.9875517923e9") -> Decimal:
    """Coulomb's Law: Fe = k * (|q1 * q2|) / r^2."""
    k_dec = Decimal(str(k))
    charge1 = Decimal(str(q1))
    charge2 = Decimal(str(q2))
    r = Decimal(str(distance_val))
    if r == Decimal('0'):
        raise ZeroDivisionError("Distance cannot be zero.")
    return k_dec * abs(charge1 * charge2) / (r ** 2)


def ohm_law_voltage(current: Any, resistance: Any) -> Decimal:
    """Ohm's Law Voltage: V = I * R."""
    return Decimal(str(current)) * Decimal(str(resistance))


def ohm_law_current(voltage: Any, resistance: Any) -> Decimal:
    """Ohm's Law Current: I = V / R."""
    r = Decimal(str(resistance))
    if r == Decimal('0'):
        raise ZeroDivisionError("Resistance cannot be zero.")
    return Decimal(str(voltage)) / r


def ohm_law_resistance(voltage: Any, current: Any) -> Decimal:
    """Ohm's Law Resistance: R = V / I."""
    i = Decimal(str(current))
    if i == Decimal('0'):
        raise ZeroDivisionError("Current cannot be zero.")
    return Decimal(str(voltage)) / i


# ==============================================================================
# 11. SCIENTIFIC CALCULATOR & PHYSICS ENGINE (FLOAT-BASED)
# ==============================================================================

def kinetic_energy(mass_val: float, velocity: float) -> float:
    """E_k = 0.5 * m * v^2 [Joules]"""
    return 0.5 * mass_val * (velocity ** 2)


def gravitational_potential_energy(mass_val: float, height: float, g: float = G_EARTH) -> float:
    """E_p = m * g * h [Joules]"""
    return mass_val * g * height


def newton_second_law(mass_val: float, acceleration: float) -> float:
    """F = m * a [Newtons]"""
    return mass_val * acceleration


def universal_gravitation(m1: float, m2: float, r: float) -> float:
    """F = G * (m1 * m2) / r^2 [Newtons]"""
    if r == 0:
        raise ZeroDivisionError("Distance r cannot be zero")
    return G_GRAVITATIONAL * (m1 * m2) / (r ** 2)


def orbital_velocity(m_central: float, r: float) -> float:
    """v = sqrt(G * M / r) [m/s]"""
    if r <= 0:
        raise ValueError("Orbital radius must be positive")
    return math.sqrt(G_GRAVITATIONAL * m_central / r)


def escape_velocity(m_central: float, r: float) -> float:
    """v_e = sqrt(2 * G * M / r) [m/s]"""
    if r <= 0:
        raise ValueError("Radius must be positive")
    return math.sqrt(2 * G_GRAVITATIONAL * m_central / r)


def coulomb_force(q1: float, q2: float, r: float) -> float:
    """F = k * |q1 * q2| / r^2 [Newtons]"""
    if r == 0:
        raise ZeroDivisionError("Distance r cannot be zero")
    k = 1.0 / (4 * math.pi * EPSILON_0)
    return k * abs(q1 * q2) / (r ** 2)


def ohms_law_voltage(current: float, resistance: float) -> float:
    """V = I * R [Volts]"""
    return current * resistance


def electric_power(voltage: float, current: float) -> float:
    """P = V * I [Watts]"""
    return voltage * current


def ideal_gas_pressure(n_moles: float, temp_kelvin: float, volume_m3: float) -> float:
    """P = n * R * T / V [Pascals]"""
    if volume_m3 == 0:
        raise ZeroDivisionError("Volume cannot be zero")
    return (n_moles * R_GAS * temp_kelvin) / volume_m3


def wave_velocity(freq: float, wavelength_m: float) -> float:
    """v = f * lambda [m/s]"""
    return freq * wavelength_m


def photon_energy(freq: float) -> float:
    """E = h * f [Joules]"""
    return H_PLANCK * freq


def relativistic_gamma(velocity: float) -> float:
    """Lorentz factor gamma = 1 / sqrt(1 - v^2 / c^2)"""
    if abs(velocity) >= C_LIGHT:
        raise ValueError("Velocity must be strictly less than c")
    beta = velocity / C_LIGHT
    return 1.0 / math.sqrt(1.0 - beta ** 2)


def mass_energy_equivalence(mass_val: float) -> float:
    """E = m * c^2 [Joules]"""
    return mass_val * (C_LIGHT ** 2)


# ==============================================================================
# 12. BERNOULLI & FLUID DYNAMICS ENGINE
# ==============================================================================

def bernoulli_constant(
    static_p: float, 
    fluid_density: float, 
    velocity: float,
    height: float = 0.0, 
    g: float = G_EARTH
) -> float:
    """Calculates Bernoulli's constant along a streamline: P + ½ ρ v² + ρ g h."""
    return static_p + 0.5 * fluid_density * velocity**2 + fluid_density * g * height


def bernoulli_pressure(
    fluid_density: float, 
    velocity: float, 
    height: float = 0.0,
    bernoulli_const: float = 0.0, 
    g: float = G_EARTH
) -> float:
    """Solves Bernoulli's equation for static pressure: P = C - ½ ρ v² - ρ g h."""
    return bernoulli_const - 0.5 * fluid_density * velocity**2 - fluid_density * g * height


def dynamic_pressure(fluid_density: float, velocity: float) -> float:
    """Dynamic pressure q = ½ ρ v² [Pa]."""
    return 0.5 * fluid_density * velocity**2


def total_pressure(static_pressure: float, fluid_density: float, velocity: float) -> float:
    """Total (stagnation) pressure = static + dynamic."""
    return static_pressure + dynamic_pressure(fluid_density, velocity)


def bernoulli_velocity_from_pressure_drop(fluid_density: float, delta_p: float) -> float:
    """Calculates fluid velocity from a pressure drop: v = sqrt(2 ΔP / ρ)."""
    if fluid_density <= 0:
        raise ValueError("Density must be positive")
    if delta_p < 0:
        raise ValueError("Pressure drop ΔP should be non-negative for real velocity")
    return math.sqrt(2.0 * delta_p / fluid_density)


def torricelli_speed(height: float, g: float = G_EARTH) -> float:
    """Torricelli's theorem: efflux speed v = sqrt(2 g h)."""
    if height < 0:
        raise ValueError("Height must be non-negative")
    return math.sqrt(2.0 * g * height)


def venturi_flow_rate(area1: float, area2: float, fluid_density: float, delta_p: float) -> float:
    """Volume flow rate Q through a Venturi tube (incompressible)."""
    if area1 <= 0 or area2 <= 0:
        raise ValueError("Areas must be positive")
    if area2 >= area1:
        raise ValueError("Venturi throat area A₂ must be smaller than A₁")
    if fluid_density <= 0:
        raise ValueError("Density must be positive")
    if delta_p < 0:
        raise ValueError("Pressure difference should be non-negative")
    beta2 = (area2 / area1) ** 2
    return area2 * math.sqrt(2.0 * delta_p / (fluid_density * (1.0 - beta2)))


def hydrostatic_pressure(fluid_density: float, depth: float, g: float = G_EARTH) -> float:
    """Hydrostatic pressure increase with depth: ΔP = ρ g h."""
    return fluid_density * g * depth


def bernoulli_head(
    static_p: float, 
    fluid_density: float, 
    velocity: float,
    height: float = 0.0, 
    g: float = G_EARTH
) -> float:
    """Hydraulic head form of Bernoulli's equation (units of length)."""
    if fluid_density <= 0 or g == 0:
        raise ValueError("Density and g must be non-zero")
    return static_p / (fluid_density * g)