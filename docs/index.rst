.. skopi documentation master file, created by
   sphinx-quickstart on Tue Feb 20 23:16:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to skopi's documentation!
=====================================
This is a python implementation of singfel.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   start
   theory
   modules
   example

Purpose of this package
=======================

psana is a relatively comprehensive package used in LCLS to fetch and analyze experiment data. It would be great if one can do simulation in a way compatible with psana. This package is create with this intention: to do single particle experiment simulation in the same style as we analyze the data.

This package is written with python 2 and is only tested in python 2 environment. Currently there is no plan to create a python 3 version. Even though I have not tested this package in python 3, I'm pretty sure it's not going to work.

Another thing worth mentioning is that: Zhaoyou Wang has also implemented a pysingfel which has already been incoperated into simex platform for which this package is created. So a little history is required to clarify this situation.

A little history
================

There are mainly three people involved in this project, pysingfel, Chun Hong Yoon, Zhaoyou Wang and me. Chun Hong Yoon is the author of the C++ version, singfel. Before I took over this project, Zhaoyou Wang, as a rotating graduate student, translated singfel into python which is the original pysingfel. pysingfel is mostly compatible with singfel since it's a translation. It works perfectly well (There are errors. But the error is exactly what singfel would have produced.). My version is not since new requirement appears: to be compatible with psana style. So when you try to use this version, you need to pay attention to this documentation.

