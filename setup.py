#
# setup.py
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
# pylint: skip-file
"""Package information of fraud_eagle package for review graph mining project.
"""
from setuptools import setup, find_packages


def _take_package_name(name):
    """Returns only package name from a given name.

    If the given name is a URL, it takes only package name. Otherwise,
    returns the given name.

    Args:
      name: a package name or URL.

    Returns:
      the associated package name.
    """
    name = name.strip()
    if name.startswith("-e"):
        name = name.split("=")[1]
        return name.split("-")[0]
    else:
        return name


def _load_requires_from_file(filepath):
    """Read a package list from a given file path.

    Args:
      filepath: file path of the package list.

    Returns:
      a list of package names.
    """
    with open(filepath) as fp:
        return [_take_package_name(pkg_name) for pkg_name in fp.readlines()]


setup(
    name='rgmining-fraud-eagle',
    version='0.9.0',
    author="Junpei Kawamoto",
    author_email="kawamoto.junpei@gmail.com",
    description="An implementation of Fraud Eagle algorithm",
    url="https://github.com/rgmining/frad-eagle",
    packages=find_packages(exclude=["tests"]),
    install_requires=_load_requires_from_file("requirements.txt"),
    dependency_links=[
        "git+https://github.com/rgmining/common.git#egg=rgmining_common-0.9.0"
    ],
    test_suite='tests.suite'
)
