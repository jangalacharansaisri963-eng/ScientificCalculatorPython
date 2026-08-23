"""
Central function registry. Modules register themselves by calling register()
or by exposing a REGISTRY / FUNCTION_REGISTRY / EULER_FUNCTIONS / MATH_LIB dict.
"""
from typing import Callable, Any, Dict, Optional

FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(
    name: Optional[str] = None,
    category: str = "General",
    desc: str = "",
) -> Callable:
    """Decorator or direct call to register a function."""

    def decorator(func: Callable) -> Callable:
        key = name or func.__name__
        for suffix in ("_func", "_function"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                break
        doc = desc or (func.__doc__ or "").strip().split("\n")[0]
        FUNCTION_REGISTRY[key] = {
            "func": func,
            "category": category,
            "desc": doc,
        }
        return func

    if callable(name):
        func = name
        name = None
        return decorator(func)
    return decorator


def register_dict(
    d: Dict[str, Any],
    category: str = "General",
    overwrite: bool = True,
) -> None:
    """
    Merge name -> callable (or name -> meta dict) into the registry.
    If overwrite is False, existing keys are left untouched.
    """
    if not isinstance(d, dict):
        return
    for key, val in d.items():
        if not overwrite and key in FUNCTION_REGISTRY:
            continue
        if callable(val):
            FUNCTION_REGISTRY[key] = {
                "func": val,
                "category": category,
                "desc": (val.__doc__ or "").strip().split("\n")[0] if hasattr(val, "__doc__") else "",
            }
        elif isinstance(val, dict) and "func" in val and callable(val["func"]):
            FUNCTION_REGISTRY[key] = {
                "func": val["func"],
                "category": val.get("category", category),
                "desc": val.get("desc", ""),
            }


def get_function(name: str) -> Optional[Callable]:
    item = FUNCTION_REGISTRY.get(name)
    return item["func"] if item else None


def list_functions(category: Optional[str] = None) -> list:
    if category is None:
        return sorted(FUNCTION_REGISTRY.keys())
    return sorted(
        k for k, v in FUNCTION_REGISTRY.items() if v.get("category") == category
    )
