#
#  graph.py
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
"""Provide a bipartite graph class implementing Fraud Eagle algorithm."""

from functools import lru_cache
from logging import getLogger
from typing import Any, Final, Optional, cast

import networkx as nx
import numpy as np

from fraud_eagle.labels import ProductLabel, ReviewLabel, UserLabel
from fraud_eagle.likelihood import psi
from fraud_eagle.prior import phi_p, phi_u

_LOGGER: Final = getLogger(__name__)
"""Logging object."""

_LOG_POINT_5: Final = float(np.log(0.5))
"""Precomputed value, the logarithm of 0.5."""


def _logaddexp(x1: float, x2: float) -> float:
    """Wrapper of np.logaddexp to solve a type problem."""
    return cast(float, np.logaddexp(x1, x2))


class Node:
    """Define a node of the bipartite graph model.

    Each node has a reference to a graph object, and has a name.
    Thus, to make a node, both of them are required.

    Args:
      graph: reference of the parent graph.
      name: name of this node.
    """

    graph: Final["ReviewGraph"]
    """Reference of the parent graph."""
    name: Final[str]
    """Name of this node."""

    __slots__ = ("graph", "name")

    def __init__(self, graph: "ReviewGraph", name: str) -> None:
        self.graph = graph
        self.name = name

    def __hash__(self) -> int:
        """Returns a hash value of this instance."""
        return 13 * hash(type(self)) + 17 * hash(self.name)

    def __str__(self) -> str:
        """Returns the name of this node."""
        return self.name


class Reviewer(Node):
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

    Args:
      graph: reference of the parent graph.
      name: name of this node.
    """

    __slots__ = ()

    @property
    def anomalous_score(self) -> float:
        """Anomalous score of this reviewer."""
        b = {}
        for u_label in iter(UserLabel):
            b[u_label] = phi_u(u_label) + self.graph.prod_message_from_products(self, None, u_label)

        return cast(float, np.exp(b[UserLabel.FRAUD] - _logaddexp(*b.values())))


class Product(Node):
    """Product node in ReviewGraph.

    Each product has a summary of its ratings. In Fraud Eagle, we uses the
    weighted average of ratings given to the product as the summary.
    The weights are anomalous scores of reviewers.

    Thus, letting :math:`r_{i}` be the rating given by :math:`i`-th reviewer,
    and :math:`a_{i}` be the anomalous score of :math:`i`-th reviewer,
    the summary of the product is defined as

    .. math::
        \\frac{\\sum_{i}a_{i}r_{i}}{\\sum_{i}a_{i}}

    Args:
      graph: reference of the parent graph.
      name: name of this node.
    """

    __slots__ = ()

    @property
    def summary(self) -> float:
        """Summary of ratings given to this product."""
        reviewers = self.graph.retrieve_reviewers(self)
        ratings = [self.graph.retrieve_review(r, self).rating for r in reviewers]
        weights = [1 - r.anomalous_score for r in reviewers]
        if sum(weights) == 0:
            return float(np.mean(ratings))
        else:
            return float(np.average(ratings, weights=weights))


class Review:
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

    Args:
      rating: the normalized rating of this review.
    """

    rating: Final[float]
    """The normalized rating of this review."""

    _user_to_product: Final[dict[ProductLabel, float]]
    _product_to_user: Final[dict[UserLabel, float]]

    __slots__ = ("rating", "_user_to_product", "_product_to_user")

    def __init__(self, rating: float) -> None:
        self.rating = rating
        self._user_to_product = {ProductLabel.GOOD: _LOG_POINT_5, ProductLabel.BAD: _LOG_POINT_5}
        self._product_to_user = {UserLabel.HONEST: _LOG_POINT_5, UserLabel.FRAUD: _LOG_POINT_5}

    @property
    def evaluation(self) -> ReviewLabel:
        """Returns a label of this review.

        If the rating is grater or equal to :math:`0.5`,
        :data:`ReviewLabel.PLUS<fraud_eagle.labels.ReviewLabel.PLUS>` is returned.
        Otherwise, :data:`ReviewLabel.MINUS<fraud_eagle.labels.ReviewLabel.MINUS>` is returned.
        """
        if self.rating >= 0.5:
            return ReviewLabel.PLUS
        else:
            return ReviewLabel.MINUS

    def user_to_product(self, label: ProductLabel) -> float:
        """Message function from the user to the product associated with this review.

        The argument `label` must be one of the {:data:`ProductLabel.GOOD<fraud_eagle.labels.ProductLabel.GOOD>`,
        :data:`ProductLabel.BAD<fraud_eagle.labels.ProductLabel.BAD>`}.

        This method returns the logarithm of the value of the message function
        for a given label.

        Args:
          label: label of the product.

        Returns:
          the logarithm of the :math:`m_{u\\rightarrow p}(label)`,
          where :math:`u` and :math:`p` is the user and the product, respectively.
        """
        return self._user_to_product[label]

    def product_to_user(self, label: UserLabel) -> float:
        """Message function from the product to the user associated with this review.

        The argument `label` must be one of the
        {:data:`UserLabel.HONEST<fraud_eagle.labels.UserLabel.HONEST>`,
        :data:`UserLabel.FRAUD<fraud_eagle.labels.UserLabel.FRAUD>`}.

        This method returns the logarithm of the value of the message function
        for a given label.

        Args:
          label: label of the user.

        Returns:
          the logarithm of the :math:`m_{p\\rightarrow u}(label)`,
          where :math:`u` and :math:`p` is the user and the product, respectively.
        """
        return self._product_to_user[label]

    def update_user_to_product(self, label: ProductLabel, value: float) -> None:
        """Update user-to-product message value.

        The argument `label` must be one of the {:data:`ProductLabel.GOOD<fraud_eagle.labels.ProductLabel.GOOD>`,
        :data:`ProductLabel.BAD<fraud_eagle.labels.ProductLabel.BAD>`}.

        Note that this method doesn't normalize any given values.

        Args:
          label: product label,
          value: new message value.
        """
        self._user_to_product[label] = value

    def update_product_to_user(self, label: UserLabel, value: float) -> None:
        """Update product-to-user message value.

        The argument `label` must be one of the
        {:data:`UserLabel.HONEST<fraud_eagle.labels.UserLabel.HONEST>`,
        :data:`UserLabel.FRAUD<fraud_eagle.labels.UserLabel.FRAUD>`}.

        Note that this method doesn't normalize any given values.

        Args:
          label: user label,
          value: new message value.
        """
        self._product_to_user[label] = value


class ReviewGraph:
    """A bipartite graph modeling reviewers and products relationships.

    Args:
        epsilon: a hyper parameter in (0, 0.5).
    """

    graph: Final[nx.DiGraph]
    """Graph object of networkx."""
    reviewers: Final[list[Reviewer]]
    """A collection of reviewers."""
    products: Final[list[Product]]
    """A collection of products."""
    epsilon: Final[float]
    """Hyper parameter."""

    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0.0 or epsilon >= 0.5:
            raise ValueError("Hyper parameter epsilon must be in (0, 0.5):", epsilon)
        self.graph = nx.DiGraph()
        self.reviewers = []
        self.products = []
        self.epsilon = epsilon

    def new_reviewer(self, name: str, *_args: Any, **_kwargs: Any) -> Reviewer:
        """Create a new reviewer and add it to this graph.

        Args:
          name: name of the new reviewer,

        Returns:
          a new reviewer.
        """
        reviewer = Reviewer(self, name)
        self.graph.add_node(reviewer)
        self.reviewers.append(reviewer)
        return reviewer

    def new_product(self, name: str) -> Product:
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

    def add_review(self, reviewer: Reviewer, product: Product, rating: float, *_args: Any, **_kwargs: Any) -> Review:
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

    @lru_cache
    def retrieve_reviewers(self, product: Product) -> list[Reviewer]:
        """Retrieve reviewers review a given product.

        Args:
          product: Product.

        Returns:
          a collection of reviewers who review the product.
        """
        return list(self.graph.predecessors(product))

    @lru_cache
    def retrieve_products(self, reviewer: Reviewer) -> list[Product]:
        """Retrieve products a given reviewer reviews.

        Args:
          reviewer: Reviewer.

        Returns:
          a collection of products the given reviewer reviews.
        """
        return list(self.graph.successors(reviewer))

    @lru_cache
    def retrieve_review(self, reviewer: Reviewer, product: Product) -> Review:
        """Retrieve a review a given reviewer posts to a given product.

        Args:
          reviewer: Reviewer,
          product: Product,

        Returns:
          a reviewer associated with the given reviewer and product.
        """
        return cast(Review, self.graph[reviewer][product]["review"])

    def update(self) -> float:
        """Update reviewers' anomalous scores and products' summaries.

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
        diffs: list[float] = []
        # Update messages from users to products.
        for reviewer in self.reviewers:
            for product in self.retrieve_products(reviewer):
                message_to_product = {}
                for p_label in iter(ProductLabel):
                    message_to_product[p_label] = self._update_user_to_product(reviewer, product, p_label)
                s = _logaddexp(*message_to_product.values())
                review = self.retrieve_review(reviewer, product)
                for p_label in iter(ProductLabel):
                    updated = message_to_product[p_label] - s
                    diffs.append(abs(np.exp(review.user_to_product(p_label)) - np.exp(updated)))
                    review.update_user_to_product(p_label, updated)

        # Update messages from products to users.
        for product in self.products:
            for reviewer in self.retrieve_reviewers(product):
                message_to_user = {}
                for u_label in iter(UserLabel):
                    message_to_user[u_label] = self._update_product_to_user(reviewer, product, u_label)
                s = _logaddexp(*message_to_user.values())
                review = self.retrieve_review(reviewer, product)
                for u_label in iter(UserLabel):
                    updated = message_to_user[u_label] - s
                    diffs.append(abs(np.exp(review.product_to_user(u_label)) - np.exp(updated)))
                    review.update_product_to_user(u_label, updated)

        histo, edges = np.histogram(diffs)
        _LOGGER.info(
            "Differentials:\n"
            + "\n".join("  {0}-{1}: {2}".format(edges[i], edges[i + 1], v) for i, v in enumerate(histo))
        )

        # Clear memoized values since messages are updated.
        self.prod_message_from_all_users.cache_clear()
        self.prod_message_from_all_products.cache_clear()

        return max(diffs)

    def _update_user_to_product(self, reviewer: Reviewer, product: Product, p_label: ProductLabel) -> float:
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
          p_label: produce label,

        Returns:
          a logarithm of the updated message from the given reviewer to the
          given product with the given product label.
        """
        review = self.retrieve_review(reviewer, product)
        res: dict[UserLabel, float] = {}
        for u_label in iter(UserLabel):
            res[u_label] = (
                np.log(psi(u_label, p_label, review.evaluation, self.epsilon))
                + phi_u(u_label)
                + self.prod_message_from_products(reviewer, product, u_label)
            )
        return _logaddexp(*res.values())

    def _update_product_to_user(self, reviewer: Reviewer, product: Product, u_label: UserLabel) -> float:
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
          u_label: user label,

        Returns:
          a logarithm of the updated message from the given product to the
          given reviewer with the given user label.
        """
        review = self.retrieve_review(reviewer, product)
        res: dict[ProductLabel, float] = {}
        for p_label in iter(ProductLabel):
            res[p_label] = (
                np.log(psi(u_label, p_label, review.evaluation, self.epsilon))
                + phi_p(p_label)
                + self.prod_message_from_users(reviewer, product, p_label)
            )
        return _logaddexp(*res.values())

    @lru_cache
    def prod_message_from_all_users(self, product: Product, p_label: ProductLabel) -> float:
        """Compute a product of messages to a product.

        This helper function computes a logarithm of the product of messages such as

        .. math::
            \\prod_{Y_{k} \\in \\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}}
            m_{k\\rightarrow j}(y_{j}),

        where :math:`\\cal{N}_{j} \\cap \\cal{Y}^{\\cal{U}}` means a set
        of reviewers who review the given product except the given reviewer,
        :math:`y_{j}` is a product label and one of the {GOOD, BAD}.

        If reviewer is None, compute a product of all messages sending to the
        product.

        Args:
          product : Product,
          p_label: product label

        Returns:
          a logarithm of the product defined above.
        """
        reviewers = set(self.retrieve_reviewers(product))
        return cast(float, np.sum([self.retrieve_review(r, product).user_to_product(p_label) for r in reviewers]))

    def prod_message_from_users(self, reviewer: Optional[Reviewer], product: Product, p_label: ProductLabel) -> float:
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
          p_label: product label

        Returns:
          a logarithm of the product defined above.
        """
        sum_all = self.prod_message_from_all_users(product, p_label)
        sum_reviewer = 0.0
        if reviewer is not None:
            sum_reviewer = self.retrieve_review(reviewer, product).user_to_product(p_label)
        return sum_all - sum_reviewer

    @lru_cache
    def prod_message_from_all_products(self, reviewer: Reviewer, u_label: UserLabel) -> float:
        """Compute a product of messages sending to a reviewer.

        This helper function computes a logarithm of the product of messages such as

        .. math::
          \\prod_{Y_{k} \\in \\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}}
          m_{k \\rightarrow i}(y_{i}),

        where :math:`\\cal{N}_{i} \\cap \\cal{Y}^{\\cal{P}}` means a set
        of products the given reviewer reviews,
        :math:`y_{i}` is a user label and one of the {HONEST, FRAUD}.

        If product is None, compute a product of all messages sending to the
        reviewer.

        Args:
          reviewer: reviewer object,
          u_label: user label.

        Returns:
          a logarithm of the product defined above.
        """
        products = set(self.retrieve_products(reviewer))
        return cast(float, np.sum([self.retrieve_review(reviewer, p).product_to_user(u_label) for p in products]))

    def prod_message_from_products(self, reviewer: Reviewer, product: Optional[Product], u_label: UserLabel) -> float:
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
          u_label: user label.

        Returns:
          a logarithm of the product defined above.
        """
        sum_all = self.prod_message_from_all_products(reviewer, u_label)
        sum_product = 0.0
        if product is not None:
            sum_product = self.retrieve_review(reviewer, product).product_to_user(u_label)

        return sum_all - sum_product
