"""
Config module for the agent exposed as an attribute-access dict
"""

# Standard
import os

# Third Party
import aconfig

_config = aconfig.Config.from_yaml(
    os.path.join(os.path.dirname(__file__), "config.yaml")
)


def __getattr__(name: str):
    return getattr(_config, name)
