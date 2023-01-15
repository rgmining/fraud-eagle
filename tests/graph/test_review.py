#
#  test_review.py
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
"""Tests for Review class.
"""
import numpy as np
from numpy.testing import assert_almost_equal

from fraud_eagle.graph import Review
from fraud_eagle.labels import ProductLabel, ReviewLabel, UserLabel


def test_evaluation() -> None:
    """Test evaluation returns collect labels."""
    r1 = Review(0.2)
    assert r1.evaluation == ReviewLabel.MINUS

    r2 = Review(0.9)
    assert r2.evaluation == ReviewLabel.PLUS


def test_update_user_to_product() -> None:
    """Test updating user-to-product message."""
    r = Review(1)
    assert_almost_equal(r.user_to_product(ProductLabel.GOOD), np.log(0.5))
    assert_almost_equal(r.user_to_product(ProductLabel.BAD), np.log(0.5))
    r.update_user_to_product(ProductLabel.GOOD, np.log(0.7))
    r.update_user_to_product(ProductLabel.BAD, np.log(0.3))
    assert_almost_equal(r.user_to_product(ProductLabel.GOOD), np.log(0.7))
    assert_almost_equal(r.user_to_product(ProductLabel.BAD), np.log(0.3))


def test_update_product_to_user() -> None:
    """Test updating product-to-user message."""
    r = Review(1)
    assert_almost_equal(r.product_to_user(UserLabel.HONEST), np.log(0.5))
    assert_almost_equal(r.product_to_user(UserLabel.FRAUD), np.log(0.5))
    r.update_product_to_user(UserLabel.HONEST, np.log(0.7))
    r.update_product_to_user(UserLabel.FRAUD, np.log(0.3))
    assert_almost_equal(r.product_to_user(UserLabel.HONEST), np.log(0.7))
    assert_almost_equal(r.product_to_user(UserLabel.FRAUD), np.log(0.3))
