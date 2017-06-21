#
# graph_test.py
#
# Copyright (c) 2016-2017 Junpei Kawamoto
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
# along with rgmining-fraud-eagle. If not, see <http://www.gnu.org/licenses/>.
#
"""Unit test for graph module in fraud_eagle package.
"""
from __future__ import division
from collections import defaultdict, OrderedDict
import random
import unittest
import networkx as nx
import numpy as np
from fraud_eagle.constants import HONEST, FRAUD, GOOD, BAD, PLUS, MINUS
from fraud_eagle.graph import Review, ReviewGraph
from fraud_eagle.likelihood import psi


class TestReview(unittest.TestCase):
    """Test case for Review class.
    """

    def test_evaluation(self):
        """Test evaluation returns collect labels.
        """
        r1 = Review(0.2)
        self.assertEqual(r1.evaluation, MINUS)
        r2 = Review(0.9)
        self.assertEqual(r2.evaluation, PLUS)

    def test_update_user_to_product(self):
        """Test updating user-to-product message.
        """
        r = Review(1)
        self.assertAlmostEqual(r.user_to_product(GOOD), np.log(0.5))
        self.assertAlmostEqual(r.user_to_product(BAD), np.log(0.5))
        r.update_user_to_product(GOOD, np.log(0.7))
        r.update_user_to_product(BAD, np.log(0.3))
        self.assertAlmostEqual(r.user_to_product(GOOD), np.log(0.7))
        self.assertAlmostEqual(r.user_to_product(BAD), np.log(0.3))
        with self.assertRaises(ValueError):
            r.user_to_product(HONEST)
        with self.assertRaises(ValueError):
            r.update_user_to_product(HONEST, 1)

    def test_update_product_to_user(self):
        """Test updating product-to-user message.
        """
        r = Review(1)
        self.assertAlmostEqual(r.product_to_user(HONEST), np.log(0.5))
        self.assertAlmostEqual(r.product_to_user(FRAUD), np.log(0.5))
        r.update_product_to_user(HONEST, np.log(0.7))
        r.update_product_to_user(FRAUD, np.log(0.3))
        self.assertAlmostEqual(r.product_to_user(HONEST), np.log(0.7))
        self.assertAlmostEqual(r.product_to_user(FRAUD), np.log(0.3))
        with self.assertRaises(ValueError):
            r.product_to_user(GOOD)
        with self.assertRaises(ValueError):
            r.update_product_to_user(GOOD, 1)


class TestEditReviewGraph(unittest.TestCase):
    """Test case for editing ReviewGraph class.
    """

    def setUp(self):
        """Set up for tests.
        """
        self.graph = ReviewGraph(0.1)

    def test_new_reviewer(self):
        """Test new reviewer has a given name.
        """
        name = "test-name"
        reviewer = self.graph.new_reviewer(name)
        self.assertEqual(reviewer.name, name)
        self.assertIn(reviewer, nx.nodes(self.graph.graph))

    def test_new_product(self):
        """Test new product has a given name.
        """
        name = "test-product"
        product = self.graph.new_product(name)
        self.assertEqual(product.name, name)
        self.assertIn(product, nx.nodes(self.graph.graph))

    def test_add_review(self):
        """Test adding a review.
        """
        reviewer = self.graph.new_reviewer("test-reviewer")
        product = self.graph.new_product("test-product")
        rating = random.randint(1, 5)
        review = self.graph.add_review(reviewer, product, rating)
        self.assertEqual(review.rating, rating)
        self.assertEqual(self.graph.graph[reviewer][product]["review"], review)


class TestReviewGraphWithASampleGraph(unittest.TestCase):
    """Test case for retriving methods and update method in ReviewGraph.

    This class sets up a small sample graph and uses it to all tests.
    """

    def setUp(self):
        """Prepare a test.

        This function creates a graph and makes a sample graph defiend as

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
        self.graph = ReviewGraph(0.1)
        self.reviewers = [
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(2)
        ]
        self.products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)
        ]
        self.reviews = defaultdict(dict)
        for i, r in enumerate(self.reviewers):
            for j in range(i, len(self.products)):
                self.reviews[i][j] = self.graph.add_review(
                    r, self.products[j], random.random())

    def test_reviewers(self):
        """Test reviewers property.
        """
        self.assertEqual(set(self.graph.reviewers), set(self.reviewers))

    def test_products(self):
        """Test products property.
        """
        self.assertEqual(set(self.graph.products), set(self.products))

    def test_retrieve_reviewers(self):
        """Test retriving reviewers from a product.
        """
        self.assertEqual(
            set(self.graph.retrieve_reviewers(self.products[0])),
            set(self.reviewers[:1]))
        self.assertEqual(
            set(self.graph.retrieve_reviewers(self.products[1])),
            set(self.reviewers))
        self.assertEqual(
            set(self.graph.retrieve_reviewers(self.products[2])),
            set(self.reviewers))
        with self.assertRaises(ValueError):
            self.graph.retrieve_reviewers(self.reviewers[0])

    def test_retrieve_products(self):
        """Test retriving proucts from a reviewer

        Sample graph used in this test is as same as
        :meth:`test_retrieve_reviewers`.
        """
        self.assertEqual(
            set(self.graph.retrieve_products(self.reviewers[0])),
            set(self.products))
        self.assertEqual(
            set(self.graph.retrieve_products(self.reviewers[1])),
            set(self.products[1:]))
        with self.assertRaises(ValueError):
            self.graph.retrieve_products(self.products[0])

    def test_retrieve_review(self):
        """Test retriving reviews from a reviewer and a product.

        Sample graph used in this test is as same as
        :meth:`test_retrieve_reviewers`.
        """
        for i, r in enumerate(self.reviewers):
            for j, p in enumerate(self.products):
                if j in self.reviews[i]:
                    self.assertEqual(
                        self.graph.retrieve_review(r, p), self.reviews[i][j])
        with self.assertRaises(ValueError):
            self.graph.retrieve_review(self.reviewers[0], self.reviewers[1])
        with self.assertRaises(ValueError):
            self.graph.retrieve_review(self.products[0], self.products[1])

    def test_update(self):
        """Test update method and check the update differences are converged.
        """
        threshold = 10**-7
        diff = 0.
        for i in range(10000):
            diff = self.graph.update()
            if diff < threshold:
                print(
                    "Update difference become smaller than {0} at iteration {1}".format(
                        threshold, i))
                return
        self.fail("Update difference didn't converged: {0}".format(diff))


class TestReviewGraph(unittest.TestCase):
    """Test case for ReviewGraph with a sample graph.
    """

    def setUp(self):
        """Prepare a test.
        """
        self.epsilon = 0.1
        self.graph = ReviewGraph(self.epsilon)

    def test_prod_message_from_users(self):
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
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(3)
        ]
        product = self.graph.new_product("product-0")
        reviews = {}
        reviews[0] = self.graph.add_review(
            reviewers[0], product, random.random())
        reviews[0].update_user_to_product(GOOD, np.log(0.4))
        reviews[0].update_user_to_product(BAD, np.log(0.6))
        reviews[1] = self.graph.add_review(
            reviewers[1], product, random.random())
        reviews[1].update_user_to_product(GOOD, np.log(0.6))
        reviews[1].update_user_to_product(BAD, np.log(0.4))
        reviews[2] = self.graph.add_review(
            reviewers[2], product, random.random())
        reviews[2].update_user_to_product(GOOD, np.log(0.8))
        reviews[2].update_user_to_product(BAD, np.log(0.2))

        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_users(
                reviewers[0], product, GOOD)),
            0.6 * 0.8)
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_users(
                reviewers[1], product, BAD)),
            0.6 * 0.2)

        # Test for giving None as the reviewer, which means considering all
        # reviewers.
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_users(None, product, GOOD)),
            0.4 * 0.6 * 0.8)
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_users(None, product, BAD)),
            0.6 * 0.4 * 0.2)

        with self.assertRaises(ValueError):
            self.graph.prod_message_from_users(reviewers[1], None, BAD)

    def test_prod_message_from_products(self):
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
        reviewer = self.graph.new_reviewer("reviewer-0")
        products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)
        ]
        reviews = {}
        reviews[0] = self.graph.add_review(
            reviewer, products[0], random.random())
        reviews[0].update_product_to_user(HONEST, np.log(0.4))
        reviews[0].update_product_to_user(FRAUD, np.log(0.6))
        reviews[1] = self.graph.add_review(
            reviewer, products[1], random.random())
        reviews[1].update_product_to_user(HONEST, np.log(0.6))
        reviews[1].update_product_to_user(FRAUD, np.log(0.4))
        reviews[2] = self.graph.add_review(
            reviewer, products[2], random.random())
        reviews[2].update_product_to_user(HONEST, np.log(0.8))
        reviews[2].update_product_to_user(FRAUD, np.log(0.2))

        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_products(
                reviewer, products[0], HONEST)), 0.6 * 0.8)
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_products(
                reviewer, products[1], FRAUD)), 0.6 * 0.2)

        # Test for giving None as the product, which means considering all
        # products.
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_products(
                reviewer, None, HONEST)),
            0.4 * 0.6 * 0.8)
        self.assertAlmostEqual(
            np.exp(self.graph.prod_message_from_products(reviewer, None, FRAUD)),
            0.6 * 0.4 * 0.2)

        with self.assertRaises(ValueError):
            self.graph.prod_message_from_products(None, products[1], FRAUD)

    def test_update_user_to_product(self):
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
        reviewer = self.graph.new_reviewer("reviewer-0")
        products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)
        ]
        reviews = {}
        reviews[0] = self.graph.add_review(reviewer, products[0], 1)
        reviews[0].update_user_to_product(GOOD, np.log(0.4))
        reviews[0].update_user_to_product(BAD, np.log(0.6))
        reviews[0].update_product_to_user(HONEST, np.log(0.3))
        reviews[0].update_product_to_user(FRAUD, np.log(0.7))
        reviews[1] = self.graph.add_review(reviewer, products[1], 0)
        reviews[1].update_product_to_user(HONEST, np.log(0.6))
        reviews[1].update_product_to_user(FRAUD, np.log(0.4))
        reviews[2] = self.graph.add_review(reviewer, products[2], 1)
        reviews[2].update_product_to_user(HONEST, np.log(0.8))
        reviews[2].update_product_to_user(FRAUD, np.log(0.2))

        # 0.6*0.8: products of other messages to the reviewer with HONEST.
        # 2.0: phi of the label (constant)
        # 0.4*0.2: products of other messages to the reviewer with FRAUD.
        ans1 = 0.6 * 0.8 * 2.0 * psi(HONEST, GOOD, PLUS, self.epsilon) \
            + 0.4 * 0.2 * 2.0 * psi(FRAUD, GOOD, PLUS, self.epsilon)
        self.assertAlmostEqual(np.exp(
            self.graph._update_user_to_product(  # pylint: disable=protected-access
                reviewer, products[0], GOOD)), ans1)

        # 0.6*0.8: products of other messages to the reviewer with HONEST.
        # 2.0: phi of the label (constant)
        # 0.4*0.2: products of other messages to the reviewer with FRAUD.
        ans2 = 0.6 * 0.8 * 2.0 * psi(HONEST, BAD, PLUS, self.epsilon) \
            + 0.4 * 0.2 * 2.0 * psi(FRAUD, BAD, PLUS, self.epsilon)
        self.assertAlmostEqual(np.exp(
            self.graph._update_user_to_product(  # pylint: disable=protected-access
                reviewer, products[0], BAD)), ans2)

    def test_update_product_to_user(self):
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
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(3)
        ]
        product = self.graph.new_product("product-0")
        reviews = {}
        reviews[0] = self.graph.add_review(reviewers[0], product, 1)
        reviews[0].update_product_to_user(HONEST, np.log(0.4))
        reviews[0].update_product_to_user(FRAUD, np.log(0.6))
        reviews[0].update_user_to_product(GOOD, np.log(0.3))
        reviews[0].update_user_to_product(BAD, np.log(0.7))
        reviews[1] = self.graph.add_review(reviewers[1], product, 0)
        reviews[1].update_user_to_product(GOOD, np.log(0.6))
        reviews[1].update_user_to_product(BAD, np.log(0.4))
        reviews[2] = self.graph.add_review(reviewers[2], product, 1)
        reviews[2].update_user_to_product(GOOD, np.log(0.8))
        reviews[2].update_user_to_product(BAD, np.log(0.2))

        # 0.6*0.8: products of other messages to the product with GOOD.
        # 2.0: phi of the label (constant)
        # 0.4*0.2: products of other messages to the product with BAD.
        ans1 = 0.6 * 0.8 * 2.0 * psi(HONEST, GOOD, PLUS, self.epsilon) \
            + 0.4 * 0.2 * 2.0 * psi(HONEST, BAD, PLUS, self.epsilon)
        self.assertAlmostEqual(np.exp(
            self.graph._update_product_to_user(  # pylint: disable=protected-access
                reviewers[0], product, HONEST)), ans1)

        # 0.6*0.8: products of other messages to the product with GOOD.
        # 2.0: phi of the label (constant)
        # 0.4*0.2: products of other messages to the product with BAD.
        ans2 = 0.6 * 0.8 * 2.0 * psi(FRAUD, GOOD, PLUS, self.epsilon) \
            + 0.4 * 0.2 * 2.0 * psi(FRAUD, BAD, PLUS, self.epsilon)
        self.assertAlmostEqual(np.exp(
            self.graph._update_product_to_user(  # pylint: disable=protected-access
                reviewers[0], product, FRAUD)), ans2)


class TestReviewer(unittest.TestCase):
    """Test case for Reviewer class.
    """

    def test_anomalous_score(self):
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
        products = [
            graph.new_product("product-{0}".format(i)) for i in range(3)
        ]
        reviews = {}
        reviews[0] = graph.add_review(reviewer, products[0], 1)
        reviews[0].update_product_to_user(HONEST, np.log(0.3))
        reviews[0].update_product_to_user(FRAUD, np.log(0.7))
        reviews[1] = graph.add_review(reviewer, products[1], 0)
        reviews[1].update_product_to_user(HONEST, np.log(0.6))
        reviews[1].update_product_to_user(FRAUD, np.log(0.4))
        reviews[2] = graph.add_review(reviewer, products[2], 1)
        reviews[2].update_product_to_user(HONEST, np.log(0.8))
        reviews[2].update_product_to_user(FRAUD, np.log(0.2))

        b_honest = 2 * 0.3 * 0.6 * 0.8
        b_fraud = 2 * 0.7 * 0.4 * 0.2
        self.assertAlmostEqual(
            reviewer.anomalous_score, b_fraud / (b_honest + b_fraud))


class TestProduct(unittest.TestCase):
    """Test case for Product class.
    """

    def test_summary(self):
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
            graph.new_reviewer("reviewer-{0}".format(i)) for i in range(3)
        ]
        product = graph.new_product("product-0")

        reviews = OrderedDict()
        reviews[0] = graph.add_review(reviewers[0], product, 1)
        reviews[0].update_user_to_product(GOOD, np.log(0.3))
        reviews[0].update_user_to_product(BAD, np.log(0.7))
        reviews[1] = graph.add_review(reviewers[1], product, 0)
        reviews[1].update_user_to_product(GOOD, np.log(0.6))
        reviews[1].update_user_to_product(BAD, np.log(0.4))
        reviews[2] = graph.add_review(reviewers[2], product, 1)
        reviews[2].update_user_to_product(GOOD, np.log(0.8))
        reviews[2].update_user_to_product(BAD, np.log(0.2))

        ratings = [review.rating for review in reviews.values()]
        weights = [1 - r.anomalous_score for r in reviewers]

        self.assertAlmostEqual(
            product.summary, np.average(ratings, weights=weights))


if __name__ == "__main__":
    unittest.main()
