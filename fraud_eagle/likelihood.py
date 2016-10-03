#
# likelihood.py
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
# pylint: disable=too-many-branches
"""Define likelihood functions.

This module defines a likelihood of a pair of user and product.
See :meth:`psi` for the detailed definition of the likelihood.
"""
from __future__ import absolute_import
from fraud_eagle.constants import PLUS, MINUS, GOOD, BAD, HONEST, FRAUD


def psi(user, product, review, epsilon):
    """Likelihood of a pair of user and product.

    The likelihood is dependent on the review of the user gives the product.
    The review is one of {+, -}. We defined constant representing "+" and "-",
    thus the review is one of the {:data:`PLUS<fraud_eagle.constants.PLUS>`,
    :data:`MINUS<fraud_eagle.constants.MINUS>`}.
    On the other hand, epsilon is a given parameter.

    The likelihood :math:`\\psi_{ij}^{s}`, where :math:`i` and :math:`j` are
    indexes of user and produce, respectively, and :math:`s` is a review
    i.e. :math:`s \\in {+, -}`, is given as following tables.

    If the review is :data:`PLUS<fraud_eagle.constants.PLUS>`,

    .. csv-table::
        :header: review: +, Product: Good, Product: Bad

        User: Honest, 1 - :math:`\\epsilon`, :math:`\\epsilon`
        User: Fraud, 2 :math:`\\epsilon`, 1 - 2 :math:`\\epsilon`

    If the review is :data:`MINUS<fraud_eagle.constants.MINUS>`,

    .. csv-table::
        :header: review: -, Product: Good, Product: Bad

        User: Honest, :math:`\\epsilon`, 1 - :math:`\\epsilon`
        User: Fraud, 1 - 2 :math:`\\epsilon`, 2 :math:`\\epsilon`

    Args:
      user: user label which must be one of the { \
        :data:`HONEST<fraud_eagle.constants.HONEST>`, \
        :data:`FRAUD<fraud_eagle.constants.FRAUD>`}.

      product: product label which must be one of the \
        {:data:`GOOD<fraud_eagle.constants.GOOD>`, \
        :data:`BAD<fraud_eagle.constants.BAD>`}.

      review: review label which must be one of the \
        {:data:`PLUS<fraud_eagle.constants.PLUS>`, \
        :data:`MINUS<fraud_eagle.constants.MINUS>`}.

      epsilon: a float parameter in :math:`[0,1]`.

    Returns:
      Float value representing a likelihood of the given values.
    """
    if review == PLUS:
        if user == HONEST:
            if product == GOOD:
                return 1 - epsilon
            elif product == BAD:
                return epsilon
        elif user == FRAUD:
            if product == GOOD:
                return 2 * epsilon
            elif product == BAD:
                return 1 - 2 * epsilon
    elif review == MINUS:
        if user == HONEST:
            if product == GOOD:
                return epsilon
            elif product == BAD:
                return 1 - epsilon
        elif user == FRAUD:
            if product == GOOD:
                return 1 - 2 * epsilon
            elif product == BAD:
                return 2 * epsilon
    raise ValueError("Given arguments are invalid.", user, product, review)
