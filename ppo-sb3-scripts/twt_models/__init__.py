#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models — modular registry of policy architectures for the parallel TWT pipeline.

Usage:
    from twt_models import get_policy, list_policies
    policy = get_policy("mlp_ppo", obs_dim=288, hidden_dim=256)

Adding a new architecture:
    1. Create twt_models/<arch_name>.py.
    2. Subclass BasePolicy, set `name = "<arch_name>"`.
    3. Decorate the class with @register_policy.
    4. Add a side-effect import at the bottom of this file.
"""

from typing import Type

from .base import BasePolicy

_REGISTRY: dict = {}


def register_policy(cls: Type[BasePolicy]):
    """Class decorator: add `cls` to the policy registry by its `name`."""
    if not issubclass(cls, BasePolicy):
        raise TypeError(f"{cls.__name__} must subclass BasePolicy")
    name = getattr(cls, "name", "base")
    if name == "base" or not isinstance(name, str) or not name:
        raise ValueError(
            f"{cls.__name__}.name must be a unique non-empty string; got '{name}'"
        )
    if name in _REGISTRY:
        raise ValueError(
            f"Policy name '{name}' is already registered "
            f"({_REGISTRY[name].__name__} vs {cls.__name__})"
        )
    _REGISTRY[name] = cls
    return cls


def get_policy(name: str, **kwargs) -> BasePolicy:
    """Instantiate a registered policy by name with constructor kwargs."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_policies() -> list:
    """Return the sorted list of registered policy names."""
    return sorted(_REGISTRY.keys())


# --- Side-effect imports register every policy in the package ---
# Add a line here for each new architecture file.
from . import mlp_ppo  # noqa: F401  (registers "mlp_ppo")
from . import lstm_ppo  # noqa: F401  (registers "lstm_ppo")
from . import pointer_ppo  # noqa: F401  (registers "pointer_ppo")
from . import transformer_pointer  # noqa: F401  (registers "transformer_pointer")
from . import random_policy  # noqa: F401  (registers "random_policy")
from . import fixed_policy  # noqa: F401  (registers "fixed_policy")
from . import (
    analytical,
)  # noqa: F401  (registers every analytical_* baseline, incl. analytical_md1)
