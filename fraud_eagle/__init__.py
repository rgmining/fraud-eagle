#
# __init__.py
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
"""An implementation of Fraud Eagle algorithm.

This algorithm has been introduced by Leman Akoglu, *et al.* in `ICWSM 2013`_.

.. _ICWSM 2013: https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/viewFile/5981/6338

"""
# TODO: Describe how to construct a graph and run analyze in the package
# document.
from __future__ import absolute_import
from fraud_eagle.graph import ReviewGraph
