#
#  test_prior.py
#
#  Copyright (c) 2016-2023 Junpei Kawamoto
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
"""Tests for prior module in fraud_eagle package.
"""
import numpy as np

from fraud_eagle.labels import UserLabel, ProductLabel
from fraud_eagle.prior import phi_u, phi_p


def test_phi_u() -> None:
    """Test with possible inputs.
    """
    assert phi_u(UserLabel.HONEST) == np.log(2)
    assert phi_u(UserLabel.FRAUD) == np.log(2)


def test_phi_p() -> None:
    """Test with possible inputs.
    """
    assert phi_p(ProductLabel.GOOD) == np.log(2)
    assert phi_p(ProductLabel.BAD) == np.log(2)
