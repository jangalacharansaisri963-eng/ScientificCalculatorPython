"""
ScientificCalculator.functions — dynamic self-registration of all public names.

1. Core math modules load first (their names win for sin, sqrt, etc.)
2. Every other module is then scanned; new public names are added without
   overwriting the core ones.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

from .registry import (
    FUNCTION_REGISTRY,
    get_function as _get_function_raw,
    register,
    register_dict,
    list_functions as _list_functions_raw,
)

_SKIP = {
    "registry",
    "library",
    "numba_bridge",
    "numba_lite_backend",
    "numba_lite",
}

# These define the canonical sin/sqrt/log/etc.
_CORE = [
    "basic_math",
    "trig",
    "hyperbolic",
    "logarithms",
    "roots",
    "factorial",
    "integers",
    "factors",
    "divisibility",
    "rationals",
    "binomial",
    "compare",
    "complex_numbers",
    "simplify",
    "derivative",
    "first_degree_equation",
    "quadratic",
    "physics",
    "Euler",
    "polynomials",
    "statistics",
    "dimensions",
    "constant_generators",
]

_loaded = False


def _collect_from_module(mod: ModuleType, modname: str, overwrite: bool) -> int:
    category = modname.replace("_", " ").title()
    before = len(FUNCTION_REGISTRY)

    for attr in ("REGISTRY", "FUNCTION_REGISTRY", "EULER_FUNCTIONS", "MATH_LIB"):
        d = getattr(mod, attr, None)
        if isinstance(d, dict) and d:
            register_dict(d, category=category, overwrite=overwrite)

    batch = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        if not overwrite and name in FUNCTION_REGISTRY:
            continue
        if name in batch:
            continue
        batch[name] = obj

    if batch:
        register_dict(batch, category=category, overwrite=overwrite)

    return len(FUNCTION_REGISTRY) - before


def load_all(verbose: bool = False) -> int:
    global _loaded
    if _loaded and len(FUNCTION_REGISTRY) > 50:
        return len(FUNCTION_REGISTRY)

    package = sys.modules[__name__]
    prefix = package.__name__ + "."
    path = getattr(package, "__path__", None)
    if not path:
        return 0

    # Discover all modules
    all_names = []
    try:
        for info in pkgutil.iter_modules(path):
            if info.name in _SKIP or info.name.startswith("_"):
                continue
            all_names.append(info.name)
    except Exception as e:
        if verbose:
            sys.stderr.write(f"  discovery error: {e}\n")

    # Pass 1 — core modules (overwrite=True so they own sin/sqrt/...)
    for name in _CORE:
        if name not in all_names:
            continue
        try:
            mod = importlib.import_module(prefix + name)
            added = _collect_from_module(mod, name, overwrite=True)
            if verbose:
                sys.stderr.write(f"  core {name} (+{added})\n")
        except Exception as e:
            if verbose:
                sys.stderr.write(f"  skip core {name}: {type(e).__name__}: {e}\n")

    # Pass 2 — everything else (no overwrite)
    for name in sorted(all_names):
        if name in _CORE:
            continue
        try:
            mod = importlib.import_module(prefix + name)
            added = _collect_from_module(mod, name, overwrite=False)
            if verbose:
                sys.stderr.write(f"  extra {name} (+{added})\n")
        except Exception as e:
            if verbose:
                sys.stderr.write(f"  skip {name}: {type(e).__name__}: {e}\n")

    _loaded = True
    return len(FUNCTION_REGISTRY)


def get_function(name: str):
    if not _loaded:
        load_all()
    return _get_function_raw(name)


def list_functions(category=None):
    if not _loaded:
        load_all()
    return _list_functions_raw(category)


__all__ = [
    "FUNCTION_REGISTRY",
    "get_function",
    "register",
    "register_dict",
    "list_functions",
    "load_all",
]
