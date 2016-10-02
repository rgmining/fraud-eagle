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
