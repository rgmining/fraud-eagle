#
#  test_edit_graph.py
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
"""Tests for editing a ReviewGraph.
"""
import random

import networkx as nx
import pytest

from fraud_eagle import ReviewGraph


@pytest.fixture
def review_graph():
    return ReviewGraph(0.1)


def test_new_reviewer(review_graph):
    """Test new reviewer has a given name.
    """
    name = "test-name"
    reviewer = review_graph.new_reviewer(name)
    assert reviewer.name == name
    assert reviewer in nx.nodes(review_graph.graph)


def test_new_product(review_graph):
    """Test new product has a given name.
    """
    name = "test-product"
    product = review_graph.new_product(name)
    assert product.name == name
    assert product in nx.nodes(review_graph.graph)


def test_add_review(review_graph):
    """Test adding a review.
    """
    reviewer = review_graph.new_reviewer("test-reviewer")
    product = review_graph.new_product("test-product")
    rating = random.randint(1, 5)
    review = review_graph.add_review(reviewer, product, rating)
    assert review.rating == rating
    assert review_graph.graph[reviewer][product]["review"] == review
