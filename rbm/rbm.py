#!/usr/bin/env python
# encoding: utf-8
"""
rbm.py

rbm.py is the fastest and easiest way to use Restricted Boltzmann
Machines (RBMs). RBMs are a class of probabilistic models that can discover
hidden patterns in your data. rbm.py provides all the necessary methods with
a pythonic interface, and moreover, all methods call blazing fast C code. The
code can also run transparently on GPU thanks to
Theano (http://deeplearning.net/software/theano/).

Created by Yann N. Dauphin on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin. All rights reserved.
"""

import sys
import os

import numpy

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def symbolic(inputs):
    """
    Wrap a symbolic method so that it can also accept concrete arguments.
    The method will be compiled with the provided inputs and stored under
    the name of the method prefixed with '__'.
        
    Parameters
    ----------
    inputs: list
        inputs to use to compile the method (i.e. theano.tensor.matrix()).

    Returns
    -------
    wrapped_method: fn
    """
    def decorator(method):
        name = "__" + method.__name__
        
        def wrapper(self, *args):
            if isinstance(args[0], T.Variable):
                return method(self, *args)
            elif hasattr(self, name):
                return getattr(self, name)(*args)
            else:
                res = method(self, *inputs)
                
                if type(res) is tuple:
                    output, updates = res
                else:
                    output, updates = res, None
                
                setattr(self, name, theano.function(inputs, output,
                    updates=updates))
                
                return getattr(self, name)(*args)
        
        return wrapper
    
    return decorator


class RBM(object):
    """
    Restricted Boltzmann Machine (RBM)
    
    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML).
    
    Examples
    ========
    
    >>> import numpy, rbm
    >>> X = numpy.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = rbm.RBM(n_hiddens=2)
    >>> model.fit(X)
    """
    def __init__(self, n_hiddens=1024,
                       W=None,
                       c=None,
                       b=None,
                       K=1,
                       epsilon=0.1,
                       n_samples=10,
                       epochs=20):
        """
        Initialize an RBM.
        
        Parameters
        ----------
        n_hiddens : int, optional
            Number of binary hidden units
        W : array-like, shape (n_visibles, n_hiddens), optional
            Weight matrix, where n_visibles in the number of visible
            units and n_hiddens is the number of hidden units.
        c : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        b : array-like, shape (n_visibles,), optional
            Biases of the visible units
        K : int, optional
            Number of MCMC steps to perform on the negative chain
            after each gradient step.
        epsilon : float, optional
            Learning rate to use during learning
        n_samples : int, optional
            Number of fantasy particles to use during learning
        epochs : int, optional
            Number of epochs to perform during learning
        """
        self.n_hiddens = n_hiddens
        self._W = theano.shared(numpy.array([[]], dtype=theano.config.floatX)
            if W == None else W)
        self._c = theano.shared(numpy.array([], dtype=theano.config.floatX)
            if c == None else c)
        self._b = theano.shared(numpy.array([], dtype=theano.config.floatX)
            if b == None else b)
        self.K = K
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.epochs = epochs
        self.h_samples = theano.shared(numpy.array([[]],
            dtype=theano.config.floatX))
        self.rng = RandomStreams(numpy.random.randint(2**30))
    
    @property
    def W(self):
        return self._W.get_value()
    
    @W.setter
    def W(self, val):
        self._W.set_value(val)
    
    @property
    def b(self):
        return self._b.get_value()
    
    @b.setter
    def b(self, val):
        self._b.set_value(val)
    
    @property
    def c(self):
        return self._c.get_value()
    
    @c.setter
    def c(self, val):
        self._c.set_value(val)
    
    @symbolic([T.matrix('v')])
    def mean_h(self, v):
        """
        Computes the probabilities P({\bf h}_j=1|{\bf v}).
        
        Parameters
        ----------
        v: array-like, shape (n_samples, n_visibles)

        Returns
        -------
        h: array-like, shape (n_samples, n_hiddens)
        """
        return T.nnet.sigmoid(T.dot(v, self._W) + self._c)
    
    @symbolic([T.matrix('v')])
    def sample_h(self, v):
        """
        Sample from the distribution P({\bf h}|{\bf v}).
        
        Parameters
        ----------
        v: array-like, shape (n_samples, n_visibles)
        
        Returns
        -------
        h: array-like, shape (n_samples, n_hiddens)
        """
        return self.rng.binomial(n=1, p=self.mean_h(v),
            dtype=theano.config.floatX)
    
    @symbolic([T.matrix('h')])
    def mean_v(self, h):
        """
        Computes the probabilities P({\bf v}_i=1|{\bf h}).
        
        Parameters
        ----------
        h: array-like, shape (n_samples, n_hiddens)
        
        Returns
        -------
        v: array-like, shape (n_samples, n_visibles)
        """
        return T.nnet.sigmoid(T.dot(h, self._W.T) + self._b)
    
    @symbolic([T.matrix('h')])
    def sample_v(self, h):
        """
        Sample from the distribution P({\bf v}|{\bf h}).
        
        Parameters
        ----------
        h: array-like, shape (n_samples, n_hiddens)
        
        Returns
        -------
        v: array-like, shape (n_samples, n_visibles)
        """
        return self.rng.binomial(n=1, p=self.mean_v(h),
            dtype=theano.config.floatX)
    
    @symbolic([T.matrix('v')])
    def free_energy(self, v):
        """
        Computes the free energy
        \mathcal{F}({\bf v}) = - \log \sum_{\bf h} e^{-E({\bf v},{\bf h})}.
        
        Parameters
        ----------
        v: array-like, shape (n_samples, n_visibles)
        
        Returns
        -------
        free_energy: array-like, shape (n_samples,)
        """
        return - T.dot(v, self._b) \
            - T.log(1. + T.exp(T.dot(v, self._W) + self._c)).sum(1)
    
    @symbolic([T.matrix('v')])
    def gibbs(self, v):
        """
        Perform one Gibbs MCMC sampling step.
        
        Parameters
        ----------
        v: array-like, shape (n_samples, n_visibles)
        
        Returns
        -------
        v_new: array-like, shape (n_samples, n_visibles)
        """
        h_ = self.sample_h(v)
        v_ = self.sample_v(h_)
        
        return v_
    
    @symbolic([T.matrix('v_pos')])
    def _fit(self, v_pos):
        """
        Adjust the parameters to maximize the likelihood of {\bf v}
        using Stochastic Maximum Likelihood (SML).
        
        Parameters
        ----------
        v_pos: array-like, shape (n_samples, n_visibles)
        """
        h_neg = self.h_samples
        for _ in range(self.K):
            v_neg = self.sample_v(h_neg)
            h_neg = self.sample_h(v_neg)
        
        cost = T.mean(self.free_energy(v_pos)) - T.mean(self.free_energy(v_neg))
        
        params = [self._W, self._b, self._c]
        gparams = T.grad(cost, params, consider_constant=[v_neg])
        
        updates = {}
        for p, gp in zip(params, gparams):
            updates[p] = p - self.epsilon * gp
        
        updates[self.h_samples] = h_neg
        
        loss = self._pseudo_likelihood(v_pos, updates)
        
        return loss, updates
    
    def _pseudo_likelihood(self, v_pos, updates):
        """
        Theano graph for the calculation of the pseudo-likelihood.
        
        Parameters
        ----------
        v_pos: array-like, shape (n_samples, n_visibles)
        updates: dict
            An index shared variable must be added to the updates.
        
        Returns
        -------
        pl: float
        """
        bit_i = theano.shared(value=0, name='bit_i')

        fe_xi = self.free_energy(v_pos)

        fe_xi_ = self.free_energy(T.set_subtensor(v_pos[:, bit_i],
            1 - v_pos[:, bit_i]))

        updates[bit_i] = (bit_i + 1) % v_pos.shape[1]
        
        return T.mean(v_pos.shape[1] * T.log(T.nnet.sigmoid(fe_xi_ - fe_xi)))
    
    def fit(self, X, verbose=False):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        """
        if self.W.shape[1] == 0:
            self.W = numpy.asarray(numpy.random.normal(0, 0.01,
                (X.shape[1], self.n_hiddens)), dtype=theano.config.floatX)
            self.c = numpy.zeros(self.n_hiddens, dtype=theano.config.floatX)
            self.b = numpy.zeros(X.shape[1], dtype=theano.config.floatX)
            self.h_samples.set_value(numpy.zeros(
                (self.n_samples, self.n_hiddens), dtype=theano.config.floatX))
        
        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = int(numpy.ceil(len(inds) / float(self.n_samples)))
        
        for epoch in range(self.epochs):
            loss = []
            for minibatch in range(n_batches):
                loss.append(self._fit(X[inds[minibatch::n_batches]]))
            
            if verbose:
                pl = numpy.mean(loss)
                
                print "Epoch %d, Pseudo-Likelihood = %.2f" % (epoch, pl)


def main():
    pass


if __name__ == '__main__':
    main()

