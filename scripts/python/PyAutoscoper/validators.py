from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class Validator(ABC):
    """Abstract base class for validators."""

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.private_name = f"_{name}"

    def __get__(self, obj: object, objtype: type[object] | None = None) -> Any:
        return getattr(obj, self.private_name)

    def __set__(self, obj: object, value: Any) -> None:
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value: Any) -> None:
        """Validate the value."""
        pass


class Path(Validator):
    """Validate that the value is a path."""

    def __init__(self, directory: bool = False, file: bool = False) -> None:
        self.directory = directory
        self.file = file
        if self.directory and self.file:
            raise ValueError("Cannot set both directory and file to True.")

    def validate(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be a string.")
        if not os.path.exists(value):
            raise ValueError(f"Expected {value!r} to exist.")
        if self.directory and not os.path.isdir(value):
            raise ValueError(f"Expected {value!r} to be a directory.")
        if self.file and not os.path.isfile(value):
            raise ValueError(f"Expected {value!r} to be a file.")


class Boolean(Validator):
    """Validate that the value is a bool."""

    def validate(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Expected {value!r} to be a bool.")


class Number(Validator):
    """Validate that the value is a number."""

    def __init__(
        self, min: float | int | None = None, max: float | int | None = None, types: list[type] | None = None
    ) -> None:
        self.min = min
        self.max = max
        self.types = (int, float) if types is None else types

    def validate(self, value: float | int) -> None:
        if not isinstance(value, self.types):
            raise TypeError(f'Expected {value!r} to be {",".join(self.types)}.')
        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")


class Integer(Number):
    """Validate that the value is an integer."""

    def __init__(self, min: int | None = None, max: int | None = None) -> None:
        super().__init__(min, max, types=(int,))


class Float(Number):
    """Validate that the value is a float."""

    def __init__(self, min: float | None = None, max: float | None = None) -> None:
        super().__init__(min, max, types=(float,))


class List(Validator):
    """Validate that the value is a list."""

    def __init__(self, size: int | None = None, types: list[type] | None = None) -> None:
        self.size = size
        self.types = types

    def validate(self, value: list[Any]) -> None:
        if not isinstance(value, list):
            raise TypeError(f"Expected {value!r} to be a list.")
        if self.size is not None and len(value) != self.size:
            raise ValueError(f"Expected {value!r} to have {self.size} elements.")
        if self.types is not None:
            for element in value:
                if not isinstance(element, self.types):
                    raise TypeError(f'Expected {element!r} to be {",".join(self.types)}.')


class FloatList(List):
    """Validate that the value is a list of floats."""

    def __init__(self, size: int | None = None) -> None:
        super().__init__(size, types=(float,))


class IntegerList(List):
    """Validate that the value is a list of integers."""

    def __init__(self, size: int | None = None) -> None:
        super().__init__(size, types=(int,))
