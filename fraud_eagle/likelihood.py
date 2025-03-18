#
#  likelihood.py
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
"""Define likelihood functions.

This module defines a likelihood of a pair of user and product.
See :meth:`psi` for the detailed definition of the likelihood.
"""

from fraud_eagle.labels import ProductLabel, ReviewLabel, UserLabel


# pylint: disable=too-many-branches
def psi(u_label: UserLabel, p_label: ProductLabel, r_label: ReviewLabel, epsilon: float) -> float:
    """Likelihood of a pair of user and product.

    The likelihood is dependent on the review of the user gives the product.
    The review is one of {+, -}. We defined constant representing "+" and "-",
    thus the review is one of the {:data:`PLUS<fraud_eagle.labels.PLUS>`,
    :data:`MINUS<fraud_eagle.labels.MINUS>`}.
    On the other hand, epsilon is a given parameter.

    The likelihood :math:`\\psi_{ij}^{s}`, where :math:`i` and :math:`j` are
    indexes of user and produce, respectively, and :math:`s` is a review
    i.e. :math:`s \\in {+, -}`, is given as following tables.

    If the review is :data:`PLUS<fraud_eagle.labels.PLUS>`,

    .. csv-table::
        :header: review: +, Product: Good, Product: Bad

        User: Honest, 1 - :math:`\\epsilon`, :math:`\\epsilon`
        User: Fraud, 2 :math:`\\epsilon`, 1 - 2 :math:`\\epsilon`

    If the review is :data:`MINUS<fraud_eagle.labels.MINUS>`,

    .. csv-table::
        :header: review: -, Product: Good, Product: Bad

        User: Honest, :math:`\\epsilon`, 1 - :math:`\\epsilon`
        User: Fraud, 1 - 2 :math:`\\epsilon`, 2 :math:`\\epsilon`

    Args:
      u_label: user label which must be one of the { \
        :data:`UserLabel.HONEST<fraud_eagle.labels.UserLabel.HONEST>`, \
        :data:`UserLabel.FRAUD<fraud_eagle.labels.UserLabel.FRAUD>`}.

      p_label: product label which must be one of the \
        {:data:`ProductLabel.GOOD<fraud_eagle.labels.ProductLabel.GOOD>`, \
        :data:`ProductLabel.BAD<fraud_eagle.labels.ProductLabel.BAD>`}.

      r_label: review label which must be one of the \
        {:data:`ReviewLabel.PLUS<fraud_eagle.labels.ReviewLabel.PLUS>`, \
        :data:`ReviewLabel.MINUS<fraud_eagle.labels.ReviewLabel.MINUS>`}.

      epsilon: a float parameter in :math:`[0,1]`.

    Returns:
      Float value representing a likelihood of the given values.
    """
    if r_label == ReviewLabel.PLUS:
        if u_label == UserLabel.HONEST:
            if p_label == ProductLabel.GOOD:
                return 1 - epsilon
            elif p_label == ProductLabel.BAD:
                return epsilon
        elif u_label == UserLabel.FRAUD:
            if p_label == ProductLabel.GOOD:
                return 2 * epsilon
            elif p_label == ProductLabel.BAD:
                return 1 - 2 * epsilon
    elif r_label == ReviewLabel.MINUS:
        if u_label == UserLabel.HONEST:
            if p_label == ProductLabel.GOOD:
                return epsilon
            elif p_label == ProductLabel.BAD:
                return 1 - epsilon
        elif u_label == UserLabel.FRAUD:
            if p_label == ProductLabel.GOOD:
                return 1 - 2 * epsilon
            elif p_label == ProductLabel.BAD:
                return 2 * epsilon
    raise ValueError("arguments are invalid")
