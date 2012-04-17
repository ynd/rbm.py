Introduction
============
_rbm.py_ is the fastest and easiest way to use Restricted Boltzmann
Machines (RBMs). RBMs are a class of probabilistic models that can discover
hidden patterns in your data. _rbm.py_ provides all the necessary methods with
a pythonic interface, and moreover, all methods call blazing fast C code. The
code can also run transparently on GPU thanks to
Theano (http://deeplearning.net/software/theano/).

Here's an example usage

    $ python
    >>> import numpy, rbm
    >>> X = numpy.array([[0, 1, 0], \
                         [0, 1, 1], \
                         [1, 0, 1], \
                         [1, 1, 1]]) # Improvised dataset
    >>> model = rbm.RBM(n_hiddens=2) # RBM with two hiddens units
    >>> model.fit(X) # Train using dataset X
    >>> model.sample_h(X) # Get hidden code
    array([[0, 0],
           [0, 1],
           [1, 1],
           [1, 0]])
    >>> model.gibbs(X) # MCMC step
    array([[0, 1, 1],
           [1, 0, 1],
           [0, 0, 1],
           [1, 0, 1]])

Authors and Contributors
========================
Yann N. Dauphin (@ynd)

Support or Contact
==================
Having trouble? Check out https://github.com/ynd/rbm.py/issues.
