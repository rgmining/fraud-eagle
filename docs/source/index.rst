An implementation of Fraud Eagle
===================================
.. raw:: html

   <div class="addthis_inline_share_toolbox"></div>

Fraud Eagle is an algorithm working on review graph, which is a bipartite graph
consisting of reviewer nodes and products nodes.
The aim of Fraud Eagle is finding fraudsters and fake reviews given by them.

This algorithm has been introduced by Leman Akoglu *et al.* in
`the Twenty-Seventh Conference on Artificial Intelligence
<http://www.aaai.org/Conferences/AAAI/aaai13.php>`_.
Their paper `Opinion Fraud Detection in Online Reviews by Network
Effects <https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/viewFile/5981/6338>`_
is available online. See it for more information.


Graph model
-------------
Fraud Eagle algorithm assumes review data are represented in a bipartite graph.
This bipartite graph has two kinds of nodes; reviewers and products.
One reviewer node and one product node are connected if the reviewer posts
a review to the product.
In other words, an edge in the graph represents a review.
Each review has a rating score.
We assume the score is normalized in 0 to 1.

Here is a sample of the bipartite graph.

.. graphviz::

   digraph bipartite {
      graph [label="Sample bipartite graph.", rankdir = LR];
      "reviewer-0";
      "reviewer-1";
      "product-0";
      "product-1";
      "product-2";
      "reviewer-0" -> "product-0" [label="0.2"];
      "reviewer-0" -> "product-1" [label="0.9"];
      "reviewer-0" -> "product-2" [label="0.6"];
      "reviewer-1" -> "product-1" [label="0.1"];
      "reviewer-1" -> "product-2" [label="0.7"];
   }


Usage
------

Construct a graph
^^^^^^^^^^^^^^^^^^
In order to run the Fraud Eagle algorithm, you need to create a bipartite graph
which represents your review data. The graph constructor is
:meth:`fraud_eagle.ReviewGraph`, which is an alias of
:class:`fraud_eagle.graph.ReviewGraph`.
It takes a parameter `epsilon` from 0 to 0.5 not including both ends.
See the original article `Opinion Fraud Detection in Online Reviews by Network
Effects <https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/viewFile/5981/6338>`_
for more details about this parameter.

You can instance the bipartite graph by

.. code-block:: python

  import fraud_eagle as feagle

  # Create a graph with a parameter `epsilon`.
  epsilon = 0.25
  graph = feagle.ReviewGraph(epsilon)


Then, you need to add reviewer nodes, product nodes, and review edges.
:meth:`new_reviewer()<fraud_eagle.graph.ReviewGraph.new_reviewer>` and
:meth:`new_product()<fraud_eagle.graph.ReviewGraph.new_product>` methods
of the graph create a reviewer node and a product node, respectively,
and add them to the graph. Both methods take one argument `name`, i.e. ID of
the node.
Note that, the names must be unique in a graph.

:meth:`add_review()<fraud_eagle.graph.ReviewGraph.add_review>` method add a
review to the graph. It takes a `reviewer`, a `product`, and a normalized
rating score which the reviewer posted to the product.
The normalized rating scores mean they must be in 0 to 1.

For example, let us assume there are two reviewers and three products
like the below.

.. graphviz::

   digraph bipartite {
      graph [label="Sample bipartite graph.", rankdir = LR];
      "reviewer-0";
      "reviewer-1";
      "product-0";
      "product-1";
      "product-2";
      "reviewer-0" -> "product-0" [label="0.2"];
      "reviewer-0" -> "product-1" [label="0.9"];
      "reviewer-0" -> "product-2" [label="0.6"];
      "reviewer-1" -> "product-1" [label="0.1"];
      "reviewer-1" -> "product-2" [label="0.7"];
   }

The graph can be constructed by the following code.

.. code-block:: python

   # Create reviewers and products.
   # Note that don't create them using fraud_eagle.graph.Reviewer and
   # fraud_eagle.graph.Product.
   reviewers = [graph.new_reviewer("reviewer-{0}".format(i)) for i in range(2)]
   products = [graph.new_product("product-{0}".format(i)) for i in range(3)]
   graph.add_review(reviewers[0], products[0], 0.2)
   graph.add_review(reviewers[0], products[1], 0.9)
   graph.add_review(reviewers[0], products[2], 0.6)
   graph.add_review(reviewers[1], products[0], 0.1)
   graph.add_review(reviewers[1], products[1], 0.7)


Analysis
^^^^^^^^^^^
:meth:`update()<fraud_eagle.graph.ReviewGraph.update>` runs one iteration of
`loopy belief propagation algorithm <https://arxiv.org/pdf/1206.0976.pdf>`_.
This method returns the amount of update in the iteration.
You need to run iterations until the amount of update becomes enough small.
It's depended to the review data and the parameter epsilon that how many
iterations are required to the amount of update becomes small.
Moreover, sometimes it won't be converged.
Thus, you should set some limitation to the iterations.

.. code-block:: python

   print("Start iterations.")
   max_iteration = 10000
   for i in range(max_iteration):

      # Run one iteration.
      diff = graph.update()
      print("Iteration %d ends. (diff=%s)", i + 1, diff)

      if diff < 10**-5: # Set 10^-5 as an acceptable small number.
          break


Result
^^^^^^^^
Each reviewer has an anomalous score which representing how the reviewer is
anomalous. The score is normalized in 0 to 1. To obtain that score,
use :meth:`anomalous_score<fraud_eagle.graph.Reviewer.anomalous_score>`
property.

The :class:`ReviewGraph<fraud_eagle.graph.ReviewGraph` has
:meth:`reviewers<fraud_eagle.graph.ReviewGraph.reviewers>` property,
which returns a collection of reviewers the graph has.
Thus, the following code outputs all reviewers' anomalous score.

.. code-block:: python

   for r in graph.reviewers:
       print(r.name, r.anomalous_score)

On the other hand, each product has a summarized ratings computed from all
reviews posted to the product according to each reviewers' anomalous score.
The summarized ratings are also normalized in 0 to 1.
:meth:`summary<fraud_eagle.graph.Product.summary>` property returns such
summarized rating.

The :class:`ReviewGraph<fraud_eagle.graph.ReviewGraph>` also has
:meth:`products<fraud_eagle.graph.ReviewGraph.products` property,
which returns a collection of products.
Thus, the following code outputs all products' summarized ratings.

.. code-block:: python

   for p in graph.products:
       print(p.name, p.summary)


API Reference
---------------
.. toctree::
  :glob:
  :maxdepth: 2

  modules/*


Indices and tables
--------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
---------
This software is released under The GNU General Public License Version 3,
see COPYING for more detail.
