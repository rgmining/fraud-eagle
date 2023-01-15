#
#  test_product.py
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
"""Tests for Product class.
"""
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_almost_equal

from fraud_eagle import ReviewGraph
from fraud_eagle.labels import ProductLabel


def test_summary() -> None:
    """Test summary property.

    In this test, we assume the following review graph,

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "product-0";
          "reviewer-0";
          "reviewer-1";
          "reviewer-2";
          "reviewer-0" -> "product-0" [label="+, m(good)=0.3, m(bad)=0.7"];
          "reviewer-1" -> "product-0" [label="-, m(good)=0.6, m(bad)=0.4"];
          "reviewer-2" -> "product-0" [label="+, m(good)=0.8, m(bad)=0.2"];
       }

    We use weighted average of ratings with anomalous scores as the weight
    for computing summary of ratings.
    """
    epsilon = 0.2
    graph = ReviewGraph(epsilon)
    reviewers = [
        graph.new_reviewer(f"reviewer-{i}") for i in range(3)
    ]
    product = graph.new_product("product-0")

    reviews = OrderedDict({
        0: graph.add_review(reviewers[0], product, 1),
        1: graph.add_review(reviewers[1], product, 0),
        2: graph.add_review(reviewers[2], product, 1),
    })
    reviews[0].update_user_to_product(ProductLabel.GOOD, np.log(0.3))
    reviews[0].update_user_to_product(ProductLabel.BAD, np.log(0.7))
    reviews[1].update_user_to_product(ProductLabel.GOOD, np.log(0.6))
    reviews[1].update_user_to_product(ProductLabel.BAD, np.log(0.4))
    reviews[2].update_user_to_product(ProductLabel.GOOD, np.log(0.8))
    reviews[2].update_user_to_product(ProductLabel.BAD, np.log(0.2))

    ratings = [review.rating for review in reviews.values()]
    weights = [1 - r.anomalous_score for r in reviewers]

    assert_almost_equal(product.summary, np.average(ratings, weights=weights))
