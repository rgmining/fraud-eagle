#
# likelihood_test.py
#
# Copyright (c) 2016-2017 Junpei Kawamoto
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
# along with rgmining-fraud-eagle. If not, see <http://www.gnu.org/licenses/>.
#
"""Unit test for likelihood module in fraud_eagle package.
"""
import unittest
from fraud_eagle.constants import PLUS, MINUS, GOOD, BAD, HONEST, FRAUD
from fraud_eagle.likelihood import psi


class TestPsi(unittest.TestCase):
    """Unit test for psi function.
    """
    def test(self):
        """Test for all possible input combinations.
        """
        for epsilon in (0.01, 0.1, 0.5):
            self.assertEqual(psi(HONEST, GOOD, PLUS, epsilon), 1-epsilon)
            self.assertEqual(psi(HONEST, GOOD, MINUS, epsilon), epsilon)
            self.assertEqual(psi(HONEST, BAD, PLUS, epsilon), epsilon)
            self.assertEqual(psi(HONEST, BAD, MINUS, epsilon), 1-epsilon)

            self.assertEqual(psi(FRAUD, GOOD, PLUS, epsilon), 2*epsilon)
            self.assertEqual(psi(FRAUD, GOOD, MINUS, epsilon), 1-2*epsilon)
            self.assertEqual(psi(FRAUD, BAD, PLUS, epsilon), 1-2*epsilon)
            self.assertEqual(psi(FRAUD, BAD, MINUS, epsilon), 2*epsilon)

        with self.assertRaises(ValueError):
            psi(HONEST, 3, PLUS, 10)


if __name__ == "__main__":
    unittest.main()
