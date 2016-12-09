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


def take_package_name(name):
    """Returns a package name.
    """
    if name.startswith("-e"):
        return name[name.find("=")+1:name.rfind("-")]
    else:
        return name.strip()


def load_requires_from_file(filepath):
    """Read a package list from a given file path.
    """
    with open(filepath) as fp:
        return [take_package_name(pkg_name) for pkg_name in fp.readlines()]


def load_links_from_file(filepath):
    """Read a package list and returns links.
    """
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if pkg_name.startswith("-e"):
                res.append(pkg_name.split(" ")[1])
    return res


setup(
    name='rgmining-fraud-eagle',
    version='0.9.4',
    author="Junpei Kawamoto",
    author_email="kawamoto.junpei@gmail.com",
    description="An implementation of Fraud Eagle algorithm",
    url="https://github.com/rgmining/frad-eagle",
    packages=find_packages(exclude=["tests"]),
    install_requires=load_requires_from_file("requirements.txt"),
    dependency_links=load_links_from_file("requirements.txt"),
    test_suite='tests.suite',
    license="GPLv3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
