#
#  labels.py
#
#  Copyright (c) 2016-2025 Junpei Kawamoto
#
#  This file is part of rgmining-fraud-eagle.
#
#  rgmining-fraud-eagle is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  rgmining-fraud-eagle is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with rgmining-fraud-eagle. If not, see <http://www.gnu.org/licenses/>.
"""Define constants used in Fraud Eagle package."""

from enum import Enum, auto
from typing import Final


class ReviewLabel(Enum):
    """Review label."""

    PLUS: Final = auto()
    """Constant representing "+" review."""
    MINUS: Final = auto()
    """Constant representing "-" review."""


class ProductLabel(Enum):
    """Product label."""

    GOOD: Final = auto()
    """Constant representing the good label for products."""
    BAD: Final = auto()
    """Constant representing the bad label for products."""


class UserLabel(Enum):
    """User label."""

    HONEST: Final = auto()
    """Constant representing the honest label for users."""
    FRAUD: Final = auto()
    """Constant representing the fraud label for users."""
