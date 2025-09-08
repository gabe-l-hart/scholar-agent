# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the common factory implementation
"""

# Standard
from typing import Any
import types

# Third Party
import aconfig
import pytest

# Local
from scholar_agent.utils import factory

## Helpers #####################################################################


class TypeOne(factory.FactoryConstructible):
    name = "one"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


class TypeTwo(factory.FactoryConstructible):
    name = "two"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


class TypeOtherOne(factory.FactoryConstructible):
    name = "one"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


def _make_dummy_module(class_obj: Any) -> types.ModuleType:
    """
    Helper to create a lightweight module containing a single class
    attribute named ``ClassName``.
    """
    mod = types.ModuleType("dummy_module")
    setattr(mod, "ClassName", class_obj)
    return mod


## Tests #######################################################################


def test_factory_happy_path():
    """Make sure that a factory works when used correctly"""
    fact = factory.Factory("Test")
    fact.register(TypeOne)
    fact.register(TypeTwo)

    inst_one = fact.construct({"type": "one"})
    assert isinstance(inst_one, TypeOne)
    assert isinstance(inst_one.config, aconfig.Config)

    inst_two = fact.construct({"type": "two", "config": {"foo": 1}})
    assert isinstance(inst_two, TypeTwo)
    assert isinstance(inst_two.config, aconfig.Config)
    assert inst_two.config.foo == 1


def test_factory_unregistered_error():
    """Make sure that asking to instantiate an unregistered type raises a
    ValueError
    """
    fact = factory.Factory("Test")
    with pytest.raises(ValueError):
        fact.construct({"type": "one"})


def test_factory_duplicate_registration():
    """Make sure that double registering a type is ok, but conflicting
    registration is not
    """
    fact = factory.Factory("Test")
    fact.register(TypeOne)
    fact.register(TypeOne)
    with pytest.raises(ValueError):
        fact.register(TypeOtherOne)


def test_factory_construct_with_instance_name():
    """Make sure that double registering a type is ok, but conflicting
    registration is not
    """
    fact = factory.Factory("Test")
    fact.register(TypeOne)

    inst_no_name = fact.construct({"type": "one"})
    assert isinstance(inst_no_name, TypeOne)
    assert inst_no_name.instance_name == TypeOne.name

    inst_name = "the-instance"
    inst_with_name = fact.construct({"type": "one"}, inst_name)
    assert isinstance(inst_with_name, TypeOne)
    assert inst_with_name.instance_name == inst_name


# --------------------------------------------------------------------
# ImportableFactory specific tests
# --------------------------------------------------------------------


def test_importable_factory_success(monkeypatch):
    """Successful dynamic import, registration and construction."""
    # Create a dummy class that satisfies the FactoryConstructible contract
    class Dummy(factory.FactoryConstructible):
        name = "dummy"

        def __init__(self, config, instance_name):
            self.config = config
            self.instance_name = instance_name

    dummy_mod = _make_dummy_module(Dummy)

    # Patch importlib.import_module to return our dummy module
    monkeypatch.setattr(
        factory.importlib,
        "import_module",
        lambda name: dummy_mod,
    )

    importable = factory.ImportableFactory("Test")
    inst_cfg = {
        "import_class": "dummy_module.ClassName",
        "type": "dummy",
        "config": {"bar": 2},
    }

    inst = importable.construct(inst_cfg)
    assert isinstance(inst, Dummy)
    assert isinstance(inst.config, aconfig.Config)
    assert inst.config.bar == 2
    assert inst.instance_name == Dummy.name


def test_importable_factory_non_string_import_class():
    """When import_class is not a string, a TypeError is raised."""
    importable = factory.ImportableFactory("Test")
    with pytest.raises(TypeError):
        importable.construct(
            {
                "import_class": 123,  # not a string
                "type": "foo",
            }
        )


def test_importable_factory_invalid_module(monkeypatch):
    """If the module cannot be imported, an ImportError is raised."""

    def fake_import(_):
        raise ImportError("cannot import")

    monkeypatch.setattr(factory.importlib, "import_module", fake_import)

    importable = factory.ImportableFactory("Test")
    with pytest.raises(ImportError, match="Module cannot be imported"):
        importable.construct(
            {
                "import_class": "nonexistent.module.Class",
                "type": "foo",
            }
        )


def test_importable_factory_missing_class(monkeypatch):
    """If the specified class is missing from the module, an ImportError."""
    # Module with no 'ClassName'
    empty_mod = types.ModuleType("empty_module")
    monkeypatch.setattr(factory.importlib, "import_module", lambda _: empty_mod)

    importable = factory.ImportableFactory("Test")
    with pytest.raises(ImportError, match="No such class"):
        importable.construct(
            {
                "import_class": "empty_module.ClassName",
                "type": "foo",
            }
        )


def test_importable_factory_invalid_subclass(monkeypatch):
    """If the imported class does not inherit FactoryConstructible, TypeError."""

    class NotFactory:
        name = "not_factory"

    mod = _make_dummy_module(NotFactory)
    monkeypatch.setattr(factory.importlib, "import_module", lambda _: mod)

    importable = factory.ImportableFactory("Test")
    with pytest.raises(TypeError, match="is not a subclass"):
        importable.construct(
            {
                "import_class": "dummy_module.ClassName",
                "type": "foo",
            }
        )
