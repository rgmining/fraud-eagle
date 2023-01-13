#
#  test_graph.py
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
"""Tests for graph module in fraud_eagle package.
"""
import random

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from fraud_eagle.constants import HONEST, FRAUD, GOOD, BAD, PLUS
from fraud_eagle.graph import ReviewGraph
from fraud_eagle.likelihood import psi

EPSILON = 0.1


@pytest.fixture
def review_graph():
    return ReviewGraph(EPSILON)


def test_prod_message_from_users(review_graph):
    """Test prod_message_from_users method.

    In this test, we assume a review graph and each messages as below:

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "reviewer-0";
          "reviewer-1";
          "reviewer-2";
          "product-0";
          "reviewer-0" -> "product-0" [label="m(good)=0.4, m(bad)=0.6"];
          "reviewer-1" -> "product-0" [label="m(good)=0.6, m(bad)=0.4"];
          "reviewer-2" -> "product-0" [label="m(good)=0.8, m(bad)=0.2"];
       }

    The prod_message_from_users method should return a value computed from
    the following equation:

    .. math::
       \\prod_{Y_{k} \\in \\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/user}
       m_{k\\rightarrow j}(y_{j}),

    where :math:`\\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/user` means a set
    of reviewers who review the given product except the given reviewer,
    :math:`y_{j}` is a product label and one of the {GOOD, BAD}.
    """
    reviewers = [
        review_graph.new_reviewer("reviewer-{0}".format(i)) for i in range(3)
    ]
    product = review_graph.new_product("product-0")
    reviews = {
        0: review_graph.add_review(reviewers[0], product, random.random()),
        1: review_graph.add_review(reviewers[1], product, random.random()),
        2: review_graph.add_review(reviewers[2], product, random.random()),
    }
    reviews[0].update_user_to_product(GOOD, np.log(0.4))
    reviews[0].update_user_to_product(BAD, np.log(0.6))
    reviews[1].update_user_to_product(GOOD, np.log(0.6))
    reviews[1].update_user_to_product(BAD, np.log(0.4))
    reviews[2].update_user_to_product(GOOD, np.log(0.8))
    reviews[2].update_user_to_product(BAD, np.log(0.2))

    assert_almost_equal(np.exp(review_graph.prod_message_from_users(reviewers[0], product, GOOD)), 0.6 * 0.8)
    assert_almost_equal(
        np.exp(review_graph.prod_message_from_users(
            reviewers[1], product, BAD)),
        0.6 * 0.2)

    # Test for giving None as the reviewer, which means considering all
    # reviewers.
    assert_almost_equal(np.exp(review_graph.prod_message_from_users(None, product, GOOD)), 0.4 * 0.6 * 0.8)
    assert_almost_equal(np.exp(review_graph.prod_message_from_users(None, product, BAD)), 0.6 * 0.4 * 0.2)


def test_prod_message_from_products(review_graph):
    """Test prod_message_from_products method.

    In this test, we assume a review graph and each messages as below:

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "reviewer-0";
          "product-0";
          "product-1";
          "product-2";
          "product-0" -> "reviewer-0" [label="m(honest)=0.4, m(fraud)=0.6"];
          "product-1" -> "reviewer-0" [label="m(honest)=0.6, m(fraud)=0.4"];
          "product-2" -> "reviewer-0" [label="m(honest)=0.8, m(fraud)=0.2"];
       }

    The prod_message_from_products method should return a value computed from
    the following equation:

    .. math::
       \\prod_{Y_{k} \\in \\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/product}
       m_{k \\rightarrow i}(y_{i}),

    where :math:`\\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/product` means a set
    of products the given reviewer reviews except the given product,
    :math:`y_{i}` is a user label and one of the {HONEST, FRAUD}.
    """
    reviewer = review_graph.new_reviewer("reviewer-0")
    products = [
        review_graph.new_product("product-{0}".format(i)) for i in range(3)
    ]
    reviews = {
        0: review_graph.add_review(reviewer, products[0], random.random()),
        1: review_graph.add_review(reviewer, products[1], random.random()),
        2: review_graph.add_review(reviewer, products[2], random.random()),
    }
    reviews[0].update_product_to_user(HONEST, np.log(0.4))
    reviews[0].update_product_to_user(FRAUD, np.log(0.6))
    reviews[1].update_product_to_user(HONEST, np.log(0.6))
    reviews[1].update_product_to_user(FRAUD, np.log(0.4))
    reviews[2].update_product_to_user(HONEST, np.log(0.8))
    reviews[2].update_product_to_user(FRAUD, np.log(0.2))

    assert_almost_equal(np.exp(review_graph.prod_message_from_products(reviewer, products[0], HONEST)), 0.6 * 0.8)
    assert_almost_equal(np.exp(review_graph.prod_message_from_products(reviewer, products[1], FRAUD)), 0.6 * 0.2)

    # Test for giving None as the product, which means considering all
    # products.
    assert_almost_equal(
        np.exp(review_graph.prod_message_from_products(reviewer, None, HONEST)),
        0.4 * 0.6 * 0.8)
    assert_almost_equal(np.exp(review_graph.prod_message_from_products(reviewer, None, FRAUD)), 0.6 * 0.4 * 0.2)


def test_update_user_to_product(review_graph):
    """Test updating a message from a user to a product.

    In this test, we assume the following review graph,

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "reviewer-0";
          "product-0";
          "product-1";
          "product-2";
          "reviewer-0" -> "product-0" [label="+, m(good)=0.4, m(bad)=0.6"];
          "product-0" -> "reviewer-0" [label="+, m(honest)=0.3, m(fraud)=0.7"];
          "product-1" -> "reviewer-0" [label="-, m(honest)=0.6, m(fraud)=0.4"];
          "product-2" -> "reviewer-0" [label="+, m(honest)=0.8, m(fraud)=0.2"];
       }

    The updated message is defined as

    .. math::
       m_{u\\rightarrow p}(y_{j}) \\leftarrow
        \\alpha_{1} \\sum_{y_{i} \\in \\cal{L}_{\\cal{U}}}
        \\psi_{ij}^{s}(y_{i}, y_{j}) \\phi^{\\cal{U}}_{i}(y_{i})
        \\prod_{Y_{k} \\in \\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/p}
        m_{k \\rightarrow i}(y_{i}),

    where :math:`y_{j} \\in {good, bad}`, and
    :math:`\\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/p` means a set of product
    the user :math:`u` reviews but except product :math:`p`.
    """
    reviewer = review_graph.new_reviewer("reviewer-0")
    products = [
        review_graph.new_product("product-{0}".format(i)) for i in range(3)
    ]
    reviews = {
        0: review_graph.add_review(reviewer, products[0], 1),
        1: review_graph.add_review(reviewer, products[1], 0),
        2: review_graph.add_review(reviewer, products[2], 1),
    }
    reviews[0].update_user_to_product(GOOD, np.log(0.4))
    reviews[0].update_user_to_product(BAD, np.log(0.6))
    reviews[0].update_product_to_user(HONEST, np.log(0.3))
    reviews[0].update_product_to_user(FRAUD, np.log(0.7))
    reviews[1].update_product_to_user(HONEST, np.log(0.6))
    reviews[1].update_product_to_user(FRAUD, np.log(0.4))
    reviews[2].update_product_to_user(HONEST, np.log(0.8))
    reviews[2].update_product_to_user(FRAUD, np.log(0.2))

    # 0.6*0.8: products of other messages to the reviewer with HONEST.
    # 2.0: phi of the label (constant)
    # 0.4*0.2: products of other messages to the reviewer with FRAUD.
    ans1 = 0.6 * 0.8 * 2.0 * psi(HONEST, GOOD, PLUS, EPSILON) + 0.4 * 0.2 * 2.0 * psi(FRAUD, GOOD, PLUS, EPSILON)
    assert_almost_equal(np.exp(
        review_graph._update_user_to_product(  # pylint: disable=protected-access
            reviewer, products[0], GOOD)), ans1)

    # 0.6*0.8: products of other messages to the reviewer with HONEST.
    # 2.0: phi of the label (constant)
    # 0.4*0.2: products of other messages to the reviewer with FRAUD.
    ans2 = 0.6 * 0.8 * 2.0 * psi(HONEST, BAD, PLUS, EPSILON) + 0.4 * 0.2 * 2.0 * psi(FRAUD, BAD, PLUS, EPSILON)
    assert_almost_equal(np.exp(
        review_graph._update_user_to_product(  # pylint: disable=protected-access
            reviewer, products[0], BAD)), ans2)


def test_update_product_to_user(review_graph):
    """Test updating a message from a product to a user.

    In this test, we assume the following review graph,

    .. graphviz::

       digraph bipartite {
          graph [rankdir = LR];
          "product-0";
          "reviewer-0";
          "reviewer-1";
          "reviewer-2";
          "product-0" -> "reviewer-0" [label="+, m(honest)=0.4, m(fraud)=0.6"];
          "reviewer-0" -> "product-0" [label="+, m(good)=0.3, m(bad)=0.7"];
          "reviewer-1" -> "product-0" [label="-, m(good)=0.6, m(bad)=0.4"];
          "reviewer-2" -> "product-0" [label="+, m(good)=0.8, m(bad)=0.2"];
       }

    The updated message is defined as

    .. math::
       m_{p\\rightarrow u}(y_{i}) \\leftarrow
       \\alpha_{3} \\sum_{y_{j} \\in \\cal{L}_{\\cal{P}}}
       \\psi_{ij}^{s}(y_{i}, y_{j}) \\phi^{\\cal{P}}_{j}(y_{j})
       \\prod_{Y_{k} \\in \\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/u}
       m_{k\\rightarrow j}(y_{j}),

    where :math:`y_{i} \\in {honest, fraud}`, and
    :math:`\\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/u` means a set of users
    who review the product :math:`p` but except user :math:`u`,
    """
    reviewers = [
        review_graph.new_reviewer("reviewer-{0}".format(i)) for i in range(3)
    ]
    product = review_graph.new_product("product-0")
    reviews = {
        0: review_graph.add_review(reviewers[0], product, 1),
        1: review_graph.add_review(reviewers[1], product, 0),
        2: review_graph.add_review(reviewers[2], product, 1),
    }
    reviews[0].update_product_to_user(HONEST, np.log(0.4))
    reviews[0].update_product_to_user(FRAUD, np.log(0.6))
    reviews[0].update_user_to_product(GOOD, np.log(0.3))
    reviews[0].update_user_to_product(BAD, np.log(0.7))
    reviews[1].update_user_to_product(GOOD, np.log(0.6))
    reviews[1].update_user_to_product(BAD, np.log(0.4))
    reviews[2].update_user_to_product(GOOD, np.log(0.8))
    reviews[2].update_user_to_product(BAD, np.log(0.2))

    # 0.6*0.8: products of other messages to the product with GOOD.
    # 2.0: phi of the label (constant)
    # 0.4*0.2: products of other messages to the product with BAD.
    ans1 = 0.6 * 0.8 * 2.0 * psi(HONEST, GOOD, PLUS, EPSILON) + 0.4 * 0.2 * 2.0 * psi(HONEST, BAD, PLUS, EPSILON)
    assert_almost_equal(np.exp(
        review_graph._update_product_to_user(  # pylint: disable=protected-access
            reviewers[0], product, HONEST)), ans1)

    # 0.6*0.8: products of other messages to the product with GOOD.
    # 2.0: phi of the label (constant)
    # 0.4*0.2: products of other messages to the product with BAD.
    ans2 = 0.6 * 0.8 * 2.0 * psi(FRAUD, GOOD, PLUS, EPSILON) + 0.4 * 0.2 * 2.0 * psi(FRAUD, BAD, PLUS, EPSILON)
    assert_almost_equal(np.exp(
        review_graph._update_product_to_user(  # pylint: disable=protected-access
            reviewers[0], product, FRAUD)), ans2)
