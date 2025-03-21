#
#  test_likelihood.py
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
"""Tests for likelihood module in fraud_eagle package."""

from fraud_eagle.labels import ProductLabel, ReviewLabel, UserLabel
from fraud_eagle.likelihood import psi


def test_psi() -> None:
    """Test for all possible input combinations."""
    for epsilon in (0.01, 0.1, 0.5):
        assert (
            psi(UserLabel.HONEST, ProductLabel.GOOD, ReviewLabel.PLUS, epsilon)
            == 1 - epsilon
        )
        assert (
            psi(
                UserLabel.HONEST, ProductLabel.GOOD, ReviewLabel.MINUS, epsilon
            )
            == epsilon
        )
        assert (
            psi(UserLabel.HONEST, ProductLabel.BAD, ReviewLabel.PLUS, epsilon)
            == epsilon
        )
        assert (
            psi(UserLabel.HONEST, ProductLabel.BAD, ReviewLabel.MINUS, epsilon)
            == 1 - epsilon
        )

        assert (
            psi(UserLabel.FRAUD, ProductLabel.GOOD, ReviewLabel.PLUS, epsilon)
            == 2 * epsilon
        )
        assert (
            psi(UserLabel.FRAUD, ProductLabel.GOOD, ReviewLabel.MINUS, epsilon)
            == 1 - 2 * epsilon
        )
        assert (
            psi(UserLabel.FRAUD, ProductLabel.BAD, ReviewLabel.PLUS, epsilon)
            == 1 - 2 * epsilon
        )
        assert (
            psi(UserLabel.FRAUD, ProductLabel.BAD, ReviewLabel.MINUS, epsilon)
            == 2 * epsilon
        )
