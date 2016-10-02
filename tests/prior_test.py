#
# prior_test.py
#
# Copyright (c) 2016 Junpei Kawamoto
#
# This file is part of rgmining-fraud-eagle.
#
# rgmining-fraud-eagle is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# rgmining-fraud-eagle is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
"""Unit test for prior module in fraud_eagle package.
"""
import unittest
import numpy as np
from fraud_eagle.constants import HONEST, FRAUD, GOOD, BAD
from fraud_eagle.prior import phi_u, phi_p


class TestPhiU(unittest.TestCase):
    """Unit test for phi_u function.
    """
    def test(self):
        """Test with possible inputs.
        """
        self.assertEqual(phi_u(HONEST), np.log(2))
        self.assertEqual(phi_u(FRAUD), np.log(2))
        with self.assertRaises(ValueError):
            phi_u(GOOD)


class TestPhiP(unittest.TestCase):
    """Unit test for phi_p function.
    """
    def test(self):
        """Test with possible inputs.
        """
        self.assertEqual(phi_p(GOOD), np.log(2))
        self.assertEqual(phi_p(BAD), np.log(2))
        with self.assertRaises(ValueError):
            phi_p(FRAUD)


if __name__ == "__main__":
    unittest.main()
