"""Test suite.
"""
from __future__ import absolute_import
import importlib
import itertools
import sys
import unittest


TESTS = (
    "tests.graph_test",
    "tests.prior_test",
    "tests.likelihood_test",
)
"""Collection of test modules."""


def suite():
    """Returns a test suite.
    """
    loader = unittest.TestLoader()
    res = unittest.TestSuite()

    for t in TESTS:
        mod = importlib.import_module(t)
        res.addTest(loader.loadTestsFromModule(mod))
    return res


def main():
    """The main function.

    Returns:
      Status code.
    """
    try:
        res = unittest.TextTestRunner(verbosity=2).run(suite())
    except KeyboardInterrupt:
        print "Test canceled."
        return -1
    else:
        return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
