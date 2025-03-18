#
#  prior.py
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
"""Define prior beliefs of users and products."""

from typing import Final

import numpy as np

from fraud_eagle.labels import ProductLabel, UserLabel

_LOG_2: Final = float(np.log(2.0))
"""Precomputed value, the logarithm of 2.0."""


def phi_u(_u_label: UserLabel) -> float:
    """Logarithm of a prior belief of a user.

    The definition is

    .. math::
       \\phi_{i}^{\\cal{U}}: \\cal{L}_{\\cal{U}} \\rightarrow \\mathbb{R}_{\\geq 0},

    where :math:`\\cal{U}` is a set of user nodes, :math:`\\cal{L}_{\\cal{U}}`
    is a set of user labels, and :math:`\\mathbb{R}_{\\geq 0}` is a set of real
    numbers grater or equals to :math:`0`.

    The implementation of this mapping is given as

    .. math::
       \\phi_{i}^{\\cal{U}}(y_{i}) \\leftarrow \\|\\cal{L}_{\\cal{U}}\\|.

    On the other hand, :math:`\\cal{L}_{\\cal{U}}` is given as {honest, fraud}.
    It means the mapping returns :math:`\\phi_{i} = 2` for any user.

    This function returns the logarithm of such :math:`\\phi_{i}`, i.e.
    :math:`\\log(\\phi_{i}(u))` for any user :math:`u`.

    Args:
      _u_label: User label.

    Returns:
      The logarithm of the prior belief of the label of the given user.
      However, it returns :math:`\\log 2` whatever the given user is.
    """
    return _LOG_2


def phi_p(_p_label: ProductLabel) -> float:
    """Logarithm of a prior belief of a product.

    The definition is

    .. math::
       \\phi_{j}^{\\cal{P}}: \\cal{L}_{\\cal{P}} \\rightarrow \\mathbb{R}_{\\geq 0},

    where :math:`\\cal{P}` is a set of produce nodes, :math:`\\cal{L}_{\\cal{P}}`
    is a set of product labels, and :math:`\\mathbb{R}_{\\geq 0}` is a set of real
    numbers grater or equals to :math:`0`.

    The implementation of this mapping is given as

    .. math::
       \\phi_{j}^{\\cal{P}}(y_{j}) \\leftarrow \\|\\cal{L}_{\\cal{P}}\\|.

    On the other hand, :math:`\\cal{L}_{\\cal{P}}` is given as {good, bad}.
    It means the mapping returns :math:`2` despite the given product.

    This function returns the logarithm of such :math:`\\phi_{j}`, i.e.
    :math:`\\log(\\phi_{j}(p))` for any product :math:`p`.

    Args:
      _p_label: Product label.

    Returns:
      The logarithm of the prior belief of the label of the given product.
      However, it returns :math:`\\log 2` whatever the given product is.
    """
    return _LOG_2
