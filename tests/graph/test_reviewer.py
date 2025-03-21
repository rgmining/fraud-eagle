#
#  test_reviewer.py
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
"""Tests for Reviewer class."""

import numpy as np
from numpy.testing import assert_almost_equal

from fraud_eagle import ReviewGraph
from fraud_eagle.labels import UserLabel


def test_anomalous_score() -> None:
    """Test anomalous_score property.

    In this test, we assume the following review graph,

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "reviewer-0";
          "product-0";
          "product-1";
          "product-2";
          "product-0" -> "reviewer-0" [label="+, m(honest)=0.3, m(fraud)=0.7"];
          "product-1" -> "reviewer-0" [label="-, m(honest)=0.6, m(fraud)=0.4"];
          "product-2" -> "reviewer-0" [label="+, m(honest)=0.8, m(fraud)=0.2"];
       }

    We use the belief of a reviewer that the reviewer is a fraud one
    to the anomalous score of the reviewer.

    The belief of user labels are defined as

    .. math::
       b(y_{i}) = \\alpha_{2} \\phi^{\\cal{U}}_{i}(y_{i})
        \\prod_{Y_{j} \\in \\cal{N}_{i} \\cap \\cal{Y}_{\\cal{P}}}
        m_{j \\rightarrow i}(y_{i}),

    where :math:`y_{i}` is a user label and one of the {honest, fraud} and
    :math:`\\cal{N}_{i} \\cap \\cal{Y}_{\\cal{P}}` means a set of products
    this reviewer reviews.
    """
    epsilon = 0.2
    graph = ReviewGraph(epsilon)
    reviewer = graph.new_reviewer("reviewer-0")
    products = [graph.new_product(f"product-{i}") for i in range(3)]
    reviews = {
        0: graph.add_review(reviewer, products[0], 1),
        1: graph.add_review(reviewer, products[1], 0),
        2: graph.add_review(reviewer, products[2], 1),
    }
    reviews[0].update_product_to_user(UserLabel.HONEST, np.log(0.3))
    reviews[0].update_product_to_user(UserLabel.FRAUD, np.log(0.7))
    reviews[1].update_product_to_user(UserLabel.HONEST, np.log(0.6))
    reviews[1].update_product_to_user(UserLabel.FRAUD, np.log(0.4))
    reviews[2].update_product_to_user(UserLabel.HONEST, np.log(0.8))
    reviews[2].update_product_to_user(UserLabel.FRAUD, np.log(0.2))

    b_honest = 2 * 0.3 * 0.6 * 0.8
    b_fraud = 2 * 0.7 * 0.4 * 0.2
    assert_almost_equal(
        reviewer.anomalous_score, b_fraud / (b_honest + b_fraud)
    )
