#
# graph.py
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
"""Provide a bipartite graph class implementing Fraud Eagle algorithm.
"""
from __future__ import absolute_import, division
import networkx as nx
import numpy as np
from fraud_eagle.constants import HONEST, FRAUD, GOOD, BAD, PLUS, MINUS
from fraud_eagle.likelihood import psi
from fraud_eagle.prior import phi_p, phi_u
from common import memoized


_LOG_POINT_5 = np.log(0.5)
"""Precomputed value, the logarithm of 0.5."""


class _Node(object):
    """Define a node of the bipartite graph model.

    Each node has a reference to a graph object, and has a name.
    Thus, to make a node, both of them are required.

    Attributes:
      graph: reference of the parent graph.
      name: name of this node.
    """
    __slots__ = ("graph", "name")

    def __init__(self, graph, name):
        """Construct of a node.

        Args:
          graph: reference of the parent graph.
          name: name of this node.
        """
        self.graph = graph
        self.name = name

    def __hash__(self):
        """Returns a hash value of this instance.
        """
        return 13 * hash(type(self)) + 17 * hash(self.name)

    def __str__(self):
        """Returns the name of this node.
        """
        return self.name


class Reviewer(_Node):
    """Reviewer node in ReviewGraph.

    Each reviewer has an anomalous_score property. In Fraud Eagle, we uses the
    belief that this reviewer is a fraud reviewer as the anomalous score.

    The belief is defined as

    .. math::
       b(y_{i}) = \\alpha_{2} \\phi^{\\cal{U}}_{i}(y_{i})
        \\prod_{Y_{j} \\in \\cal{N}_{i} \\cap \\cal{Y}_{\\cal{P}}}
        m_{j \\rightarrow i}(y_{i}),

    where :math:`y_{i}` is a user label and one of the {honest, fraud} and
    :math:`\\cal{N}_{i} \\cap \\cal{Y}_{\\cal{P}}` means a set of products
    this reviewer reviews. :math:`\\alpha_{2}` is a normalize constant so that
    :math:`b(honest) + b(fraud) = 1`.

    Thus, we use :math:`b(fraud)` as the anomalous score.
    """
    __slots__ = ()

    @property
    def anomalous_score(self):
        """Anomalous score of this reviewer.
        """
        b = {}
        for ulabel in (HONEST, FRAUD):
            b[ulabel] = phi_u(ulabel) \
                + self.graph.prod_message_from_products(self, None, ulabel)
        return np.exp(b[FRAUD] - np.logaddexp(*b.values()))


class Product(_Node):
    """Product node in ReviewGraph.

    Each product has a summary of its ratings. In Fraud Eagle, we uses the
    weighted average of ratings given to the product as the summary.
    The weights are anomalous scores of reviewers.

    Thus, letting :math:`r_{i}` be the rating given by :math:`i`-th reviewer,
    and :math:`a_{i}` be the anomalous score of :math:`i`-th reviewer,
    the summary of the product is defined as

    .. math::
        \\frac{\\sum_{i}a_{i}r_{i}}{\\sum_{i}a_{i}}

    """
    __slots__ = ()

    @property
    def summary(self):
        """Summary of ratings given to this product.
        """
        reviewers = self.graph.retrieve_reviewers(self)
        ratings = [self.graph.retrieve_review(
            r, self).rating for r in reviewers]
        weights = [r.anomalous_score for r in reviewers]
        if sum(weights) == 0:
            return np.mean(ratings)
        else:
            return np.average(ratings, weights=weights)


class Review(object):
    """Review represents a edge in the bipartite graph.

    Review is an edge in the bipartite graph connecting a user to a product
    if the user reviews the product. The review has a score the user gives
    to the product. Additionally, in Fraud Eagle, each review has two message
    functions, i.e. message from the user to the product, vise versa.
    Each message function takes only two values.
    For example, the message from the user to the product can take {good, bad}.

    To implement those message functions, this review class maintain four values
    associated with each function and each input. But also provide message
    functions as methods.

    Review also has a rating score given a user to a product. We assume this
    score is normalized in :math:`[0, 1]`. Fraud Eagle treats this score as a
    binary value i.e. + or -. To implement it, we choose a threshold 0.5 to
    decide each rating belonging to + group or - group, and evaluation property
    returns this label. In other words, for a review *r*,

    .. math::

       r.evaluation =
           \\begin{cases}
               PLUS \\quad (r.rating \\geq 0.5) \\\\
               MINUS \\quad (otherwise)
           \\end{cases}

    Attributes:
      rating: the normalized rating of this review.
    """
    __slots__ = ("_user_to_product", "_product_to_user", "rating")

    def __init__(self, rating):
        self._user_to_product = {GOOD: _LOG_POINT_5, BAD: _LOG_POINT_5}
        self._product_to_user = {HONEST: _LOG_POINT_5, FRAUD: _LOG_POINT_5}
        self.rating = rating

    @property
    def evaluation(self):
        """Returns a label of this review.

        If the rating is grater or equal to :math:`0.5`,
        :data:`PLUS<fraud_eagle.constants.PLUS>` is returned.
        Otherwise, :data:`MINUS<fraud_eagle.constants.MINUS>` is returned.
        """
        if self.rating >= 0.5:
            return PLUS
        else:
            return MINUS

    def user_to_product(self, label):
        """Message function from the user to the product associated with this review.

        The argument `label` must be one of the {:data:`GOOD<fraud_eagle.constants.GOOD>`,
        :data:`BAD<fraud_eagle.constants.BAD>`}.

        This method returns the logarithm of the value of the message function
        for a given label.

        Args:
          label: label of the product.

        Returns:
          the logarithm of the :math:`m_{u\\rightarrow p}(label)`,
          where :math:`u` and :math:`p` is the user and the product, respectively.

        Raises:
          ValueError: if the given label isn't one of {GOOD, BAD}.
        """
        if label not in (GOOD, BAD):
            raise ValueError("Given label isn't for a product:", label)
        return self._user_to_product[label]

    def product_to_user(self, label):
        """Message function from the product to the user associated with this review.

        The argument `label` must be one of the
        {:data:`HONEST<fraud_eagle.constants.HONEST>`,
        :data:`FRAUD<fraud_eagle.constants.FRAUD>`}.

        This method returns the logarithm of the value of the message function
        for a given label.

        Args:
          label: label of the user.

        Returns:
          the logarithm of the :math:`m_{p\\rightarrow u}(label)`,
          where :math:`u` and :math:`p` is the user and the product, respectively.

        Raises:
          ValueError: if the given label isn't one of {HONEST, FRAUD}.
        """
        if label not in (HONEST, FRAUD):
            raise ValueError("Given label isn't for a user:", label)
        return self._product_to_user[label]

    def update_user_to_product(self, label, value):
        """Update user-to-product message value.

        The argument `label` must be one of the {:data:`GOOD<fraud_eagle.constants.GOOD>`,
        :data:`BAD<fraud_eagle.constants.BAD>`}.

        Note that this method doesn't normalize any given values.

        Args:
          label: product label,
          value: new message value.

        Raises:
          ValueError: if the given label isn't one of {GOOD, BAD}.
        """
        if label not in (GOOD, BAD):
            raise ValueError("Given label isn't for a product:", label)
        self._user_to_product[label] = value

    def update_product_to_user(self, label, value):
        """Update product-to-user message value.

        The argument `label` must be one of the
        {:data:`HONEST<fraud_eagle.constants.HONEST>`,
        :data:`FRAUD<fraud_eagle.constants.FRAUD>`}.

        Note that this method doesn't normalize any given values.

        Args:
          label: user label,
          value: new message value.

        Raises:
          ValueError: if the given label isn't one of {HONEST, FRAUD}.
        """
        if label not in (HONEST, FRAUD):
            raise ValueError("Given label isn't for a user:", label)
        self._product_to_user[label] = value


class ReviewGraph(object):
    """A bipartite graph modeling reviewers and products relationships.

    Attributes:
      graph: Graph object of networkx.
      reviewers: A collection of reviewers.
      products: A collection of products.
      epsilon: Hyper parameter.
    """

    def __init__(self, epsilon):
        if epsilon <= 0. or epsilon >= 0.5:
            raise ValueError(
                "Hyper parameter epsilon must be in (0, 0.5):", epsilon)
        self.graph = nx.DiGraph()
        self.reviewers = []
        self.products = []
        self.epsilon = epsilon

    def new_reviewer(self, name, anomalous=None):  # pylint: disable=unused-argument
        """Create a new reviewer and add it to this graph.

        Args:
          name: name of the new reviewer,
          _anomalous: default anomalous score (not used in this method).

        Returns:
          a new reviewer.
        """
        reviewer = Reviewer(self, name)
        self.graph.add_node(reviewer)
        self.reviewers.append(reviewer)
        return reviewer

    def new_product(self, name):
        """Create a new product and add it to this graph.

        Args:
          name: name of the new product.

        Returns:
          a new product.
        """
        product = Product(self, name)
        self.graph.add_node(product)
        self.products.append(product)
        return product

    def add_review(self, reviewer, product, rating, _time=None):
        """Add a review from a given reviewer to a product.

        Args:
          reviewer: reviewer of the review,
          product: product of the review,
          rating: rating score of the review.

        Returns:
          a new review.
        """
        review = Review(rating)
        self.graph.add_edge(reviewer, product, review=review)
        return review

    @memoized
    def retrieve_reviewers(self, product):
        """Retrieve reviewers review a given product.

        Args:
          product: Product.

        Returns:
          a collection of reviewers who review the product.

        Raises:
          ValueError: if the given product isn't an instance of Product.
        """
        if not isinstance(product, Product):
            raise ValueError(
                "Given product isn't an instance of Product:", product)
        return self.graph.predecessors(product)

    @memoized
    def retrieve_products(self, reviewer):
        """Retrieve products a given reviewer reviews.

        Args:
          reviewer: Reviewer.

        Returns:
          a collection of products the given reviewer reviews.

        Raises:
          ValueError: if the given reviewer isn't an instance of Reviewer.
        """
        if not isinstance(reviewer, Reviewer):
            raise ValueError(
                "Given reviewer isn't an instance of Reviewer:", reviewer)
        return self.graph.successors(reviewer)

    @memoized
    def retrieve_review(self, reviewer, product):
        """Retrieve a review a given reviewer posts to a given product.

        Args:
          reviewer: Reviewer,
          product: Product,

        Returns:
          a reviewer associated with the given reviewer and product.

        Raises:
          ValueError: if the given reviewer isn't an instance of Reviewer or
            the given product isn't an instance of Product.
        """
        if not isinstance(reviewer, Reviewer):
            raise ValueError(
                "Given reviewer isn't an instance of Reviwer:", reviewer)
        elif not isinstance(product, Product):
            raise ValueError(
                "Given product isn't an instance of Product:", product)
        return self.graph[reviewer][product]["review"]

    def update(self):
        """ Update reviewers' anomalous scores and products' summaries.

        For each user :math:`u`, update messages to every product :math:`p`
        the user reviews. The message function :math:`m_{u\\rightarrow p}`
        takes one argument i.e. label of the receiver product.
        The label is one of {good, bad}.
        Therefore, we need to compute updated :math:`m_{u\\rightarrow p}(good)`
        and :math:`m_{u\\rightarrow p}(bad)`.

        The updated messages are defined as

        .. math::
           m_{u\\rightarrow p}(y_{j}) \\leftarrow
            \\alpha_{1} \\sum_{y_{i} \\in \\cal{L}_{\\cal{U}}}
            \\psi_{ij}^{s}(y_{i}, y_{j}) \\phi^{\\cal{U}}_{i}(y_{i})
            \\prod_{Y_{k} \\in \\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/p}
            m_{k \\rightarrow i}(y_{i}),

        where :math:`y_{j} \\in {good, bad}`, and
        :math:`\\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/p` means a set of product
        the user :math:`u` reviews but except product :math:`p`.

        For each product :math:`p`, update message to every user :math:`u`
        who reviews the product. The message function :math:`m_{p\\rightarrow u}`
        takes one argument i.e. label of the receiver user.
        The label is one of {honest, fraud}.
        Thus, we need to compute updated :math:`m_{p\\rightarrow u}(honest)`
        and :math:`m_{p\\rightarrow u}(fraud)`.

        The updated messages are defined as

        .. math::
           m_{p\\rightarrow u}(y_{i}) \\leftarrow
           \\alpha_{3} \\sum_{y_{j} \\in \\cal{L}_{\\cal{P}}}
           \\psi_{ij}^{s}(y_{i}, y_{j}) \\phi^{\\cal{P}}_{j}(y_{j})
           \\prod_{Y_{k} \\in \\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/u}
           m_{k\\rightarrow j}(y_{j}),

        where :math:`y_{i} \\in {honest, fraud}`, and
        :math:`\\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/u` means a set of users
        who review the product :math:`p` but except user :math:`u`,

        This method runs one iteration of update for both reviewers, i.e. users
        and products. It returns the maximum difference between an old message
        value and the associated new message value. You can stop iteration when
        the update gap reaches satisfied small value.

        Returns:
          maximum difference between an old message value and its updated new
          value.
        """
        diff = -1
        # Update messages from users to products.
        for reviewer in self.reviewers:
            for product in self.retrieve_products(reviewer):
                new = {}
                for plabel in (GOOD, BAD):
                    new[plabel] = self._update_user_to_product(
                        reviewer, product, plabel)
                s = np.logaddexp(*new.values())
                review = self.retrieve_review(reviewer, product)
                for plabel in (GOOD, BAD):
                    updated = new[plabel] - s
                    diff = max(diff, abs(
                        np.exp(review.user_to_product(plabel)) - np.exp(updated)))
                    review.update_user_to_product(plabel, updated)

        # Update messages from products to users.
        for product in self.products:
            for reviewer in self.retrieve_reviewers(product):
                new = {}
                for ulabel in (HONEST, FRAUD):
                    new[ulabel] = self._update_product_to_user(
                        reviewer, product, ulabel)
                s = np.logaddexp(*new.values())
                review = self.retrieve_review(reviewer, product)
                for ulabel in (HONEST, FRAUD):
                    updated = new[ulabel] - s
                    diff = max(diff, abs(
                        np.exp(review.product_to_user(ulabel)) - np.exp(updated)))
                    review.update_product_to_user(ulabel, updated)

        return diff

    def _update_user_to_product(self, reviewer, product, plabel):
        """Compute an updated message from a user to a product with a product label.

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
        The :math:`\\psi_{ij}^{s}(y_{i}, y_{j})` takes :math:`\\epsilon` as a
        hyper parameter, self.epsilon is used for it.

        This method returns a logarithm of the updated message.

        Args:
          reviewer: Reviewer,
          product: Product,
          plabel: produce label,

        Returns:
          a logarithm of the updated message from the given reviewer to the
          given product with the given product label.
        """
        review = self.retrieve_review(reviewer, product)
        res = {}
        for ulabel in (HONEST, FRAUD):
            res[ulabel] = \
                np.log(psi(ulabel, plabel, review.evaluation, self.epsilon)) \
                + phi_u(ulabel) \
                + self.prod_message_from_products(reviewer, product, ulabel)
        return np.logaddexp(*res.values())

    def _update_product_to_user(self, reviewer, product, ulabel):
        """Compute an updated message from a product to a user with a user label.

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
        The :math:`\\psi_{ij}^{s}(y_{i}, y_{j})` takes :math:`\\epsilon` as a
        hyper parameter, self.epsilon is used for it.

        This method returns a logarithm of the updated message.

        Args:
          reviewer: Reviewer i.e. a user,
          product: Product,
          ulabel: user label,

        Returns:
          a logarithm of the updated message from the given product to the
          given reviewer with the given user label.
        """
        review = self.retrieve_review(reviewer, product)
        res = {}
        for plabel in (GOOD, BAD):
            res[plabel] = \
                np.log(psi(ulabel, plabel, review.evaluation, self.epsilon)) \
                + phi_p(plabel) \
                + self.prod_message_from_users(reviewer, product, plabel)
        return np.logaddexp(*res.values())

    def prod_message_from_users(self, reviewer, product, plabel):
        """Compute a product of messages to a product except from a reviewer.

        This helper function computes a logarithm of the product of messages such as

        .. math::
           \\prod_{Y_{k} \\in \\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/user}
           m_{k\\rightarrow j}(y_{j}),

        where :math:`\\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}/user` means a set
        of reviewers who review the given product except the given reviewer,
        :math:`y_{j}` is a product label and one of the {GOOD, BAD}.

        If reviewer is None, compute a product of all messages sending to the
        product.

        Args:
          reviewer: Reviewer, can be None,
          product : Product,
          plabel: product label

        Returns:
          a logarithm of the product defined above.
        """
        reviewers = set(self.retrieve_reviewers(product)) - set([reviewer])
        return np.sum([
            self.retrieve_review(r, product).user_to_product(plabel)
            for r in reviewers
        ])

    def prod_message_from_products(self, reviewer, product, ulabel):
        """Compute a product of messages sending to a reviewer except from a product.

        This helper function computes a logarithm of the product of messages such as

        .. math::
           \\prod_{Y_{k} \\in \\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/product}
           m_{k \\rightarrow i}(y_{i}),

        where :math:`\\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}/product` means a set
        of products the given reviewer reviews except the given product,
        :math:`y_{i}` is a user label and one of the {HONEST, FRAUD}.

        If product is None, compute a product of all messages sending to the
        reviewer.

        Args:
          reviewer: reviewer object,
          product: product object, can be None,
          ulabel: user label.

        Returns:
          a logarithm of the product defined above.
        """
        products = set(self.retrieve_products(reviewer)) - set([product])
        return np.sum([
            self.retrieve_review(reviewer, p).product_to_user(ulabel)
            for p in products
        ])
