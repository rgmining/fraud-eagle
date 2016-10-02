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
