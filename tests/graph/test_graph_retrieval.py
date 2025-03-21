#
#  test_graph_retrieval.py
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
"""Tests for retrieving methods and update method in ReviewGraph.

This class sets up a small sample graph and uses it to all tests.
"""

from collections import defaultdict
from dataclasses import dataclass
from random import random

import pytest

from fraud_eagle import ReviewGraph
from fraud_eagle.graph import Review


@dataclass
class GraphFixture:
    graph: ReviewGraph
    reviewers: list
    products: list
    reviews: dict


@pytest.fixture
def review_graph() -> GraphFixture:
    """Returns a graph and makes a sample graph defined as

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "reviewer-0";
          "reviewer-1";
          "product-0";
          "product-1";
          "product-2";
          "reviewer-0" -> "product-0";
          "reviewer-0" -> "product-1";
          "reviewer-0" -> "product-2";
          "reviewer-1" -> "product-1";
          "reviewer-1" -> "product-2";
       }

    """
    graph = ReviewGraph(0.1)
    reviewers = [graph.new_reviewer(f"reviewer-{i}") for i in range(2)]
    products = [graph.new_product(f"product-{i}") for i in range(3)]
    reviews = defaultdict[int, dict[int, Review]](dict)
    for i, r in enumerate(reviewers):
        for j in range(i, len(products)):
            reviews[i][j] = graph.add_review(r, products[j], random())
    return GraphFixture(graph, reviewers, products, reviews)


def test_reviewers(review_graph: GraphFixture) -> None:
    """Test reviewers' property."""
    assert set(review_graph.graph.reviewers) == set(review_graph.reviewers)


def test_products(review_graph: GraphFixture) -> None:
    """Test products' property."""
    assert set(review_graph.graph.products) == set(review_graph.products)


def test_retrieve_reviewers(review_graph: GraphFixture) -> None:
    """Test retrieving reviewers from a product."""
    assert set(
        review_graph.graph.retrieve_reviewers(review_graph.products[0])
    ) == set(review_graph.reviewers[:1])
    assert set(
        review_graph.graph.retrieve_reviewers(review_graph.products[1])
    ) == set(review_graph.reviewers)
    assert set(
        review_graph.graph.retrieve_reviewers(review_graph.products[2])
    ) == set(review_graph.reviewers)


def test_retrieve_products(review_graph: GraphFixture) -> None:
    """Test retrieving products from a reviewer

    Sample graph used in this test is as same as
    :meth:`test_retrieve_reviewers`.
    """
    assert set(
        review_graph.graph.retrieve_products(review_graph.reviewers[0])
    ) == set(review_graph.products)
    assert set(
        review_graph.graph.retrieve_products(review_graph.reviewers[1])
    ) == set(review_graph.products[1:])


def test_retrieve_review(review_graph: GraphFixture) -> None:
    """Test retrieving reviews from a reviewer and a product.

    Sample graph used in this test is as same as
    :meth:`test_retrieve_reviewers`.
    """
    for i, r in enumerate(review_graph.reviewers):
        for j, p in enumerate(review_graph.products):
            if j in review_graph.reviews[i]:
                assert (
                    review_graph.graph.retrieve_review(r, p)
                    == review_graph.reviews[i][j]
                )


def test_update(review_graph: GraphFixture) -> None:
    """Test update method and check the update differences are converged."""
    threshold = 10**-7
    diff = 0.0
    for i in range(10000):
        diff = review_graph.graph.update()
        if diff < threshold:
            print(
                f"Update difference become smaller than {threshold} at iteration {i}"
            )
            return
    pytest.fail(f"Update difference didn't converged: {diff}")
