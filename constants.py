"""
Mathematical and physical constants for ScientificCalculator.
"""
import math

# Mathematical constants
PI = math.pi
E = math.e
TAU = math.tau
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
LN2 = math.log(2)
LN10 = math.log(10)
INF = float('inf')
NAN = float('nan')

# Physical constants (SI units)
# Speed of light in vacuum (m/s)
C_LIGHT = 299792458
# Gravitational constant (N m^2 / kg^2)
G_GRAVITATIONAL = 6.67430e-11
# Planck's constant (J s)
H_PLANCK = 6.62607015e-34
# Reduced Planck's constant (J s)
HBAR = H_PLANCK / (2 * math.pi)
# Boltzmann constant (J / K)
K_BOLTZMANN = 1.380649e-23
# Elementary charge (C)
E_CHARGE = 1.602176634e-19
# Vacuum permittivity (F / m)
EPSILON_0 = 8.8541878128e-12
# Vacuum permeability (H / m)
MU_0 = 1.25663706212e-6
# Avogadro constant (mol^-1)
N_AVOGADRO = 6.02214076e23
# Ideal gas constant (J / (mol K))
R_GAS = 8.31446261815324
# Stefan-Boltzmann constant (W / (m^2 K^4))
SIGMA_SB = 5.670374419e-8
# Electron mass (kg)
MASS_ELECTRON = 9.1093837015e-31
# Proton mass (kg)
MASS_PROTON = 1.67262192369e-27
# Neutron mass (kg)
MASS_NEUTRON = 1.67492749804e-27
# Standard acceleration of gravity on Earth (m / s^2)
G_EARTH = 9.80665
# Standard atmospheric pressure (Pa)
ATMOSPHERE = 101325.0
# Rydberg constant (m^-1)
RYDBERG_CONST = 10973731.568160
# Fine structure constant (dimensionless)
FINE_STRUCTURE = 7.2973525693e-3

CONSTANTS_DICT = {
    'pi': PI,
    'PI': PI,
    'e': E,
    'E': E,
    'tau': TAU,
    'TAU': TAU,
    'phi': PHI,
    'PHI': PHI,
    'euler': EULER_MASCHERONI,
    'gamma_constant': EULER_MASCHERONI,
    'sqrt2': SQRT2,
    'sqrt3': SQRT3,
    'ln2': LN2,
    'ln10': LN10,
    'c': C_LIGHT,
    'G': G_GRAVITATIONAL,
    'h': H_PLANCK,
    'hbar': HBAR,
    'kb': K_BOLTZMANN,
    'kB': K_BOLTZMANN,
    'q_e': E_CHARGE,
    'e_charge': E_CHARGE,
    'eps0': EPSILON_0,
    'mu0': MU_0,
    'Na': N_AVOGADRO,
    'N_A': N_AVOGADRO,
    'R': R_GAS,
    'sigma': SIGMA_SB,
    'm_e': MASS_ELECTRON,
    'm_p': MASS_PROTON,
    'm_n': MASS_NEUTRON,
    'g': G_EARTH,
    'atm': ATMOSPHERE,
}
