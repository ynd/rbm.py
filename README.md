rbm.py
======

A pythonic library for Restricted Boltzmann Machines. This is
for people who want to give RBMs a quick try and for people
who want to understand how they are implemented. For this
purpose I tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Examples
========

    >>> import numpy, rbm
    >>> X = numpy.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = rbm.RBM(n_hiddens=2)
    >>> model.fit(X)

