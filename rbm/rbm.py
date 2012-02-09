#!/usr/bin/env python
# encoding: utf-8
"""
rbm.py

A pythonic library for Restricted Boltzmann Machines. This is
for people who want to give RBMs a quick try and for people
who want to understand how they are implemented. For this
purpose I tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Created by Yann N. Dauphin on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin. All rights reserved.
"""

import sys
import os

import numpy


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
        epsilon : float, optional
            Learning rate to use during learning
        n_samples : int, optional
            Number of fantasy particles to use during learning
        epochs : int, optional
            Number of epochs to perform during learning
        """
        self.n_hiddens = n_hiddens
        self.W = W
        self.c = c
        self.b = b
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.epochs = epochs
        self.h_samples = None
    
    def _sigmoid(self, x):
        """
        Implements the logistic function.
        
        Parameters
        ----------
        x: array-like, shape (M, N)

        Returns
        -------
        x_new: array-like, shape (M, N)
        """
        return 1. / (1. + numpy.exp(-x)) 
    
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
        return self._sigmoid(numpy.dot(v, self.W) + self.c)
    
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
        return numpy.random.binomial(1, self.mean_h(v))
    
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
        return self._sigmoid(numpy.dot(h, self.W.T) + self.b)
    
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
        return numpy.random.binomial(1, self.mean_v(h))
    
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
        return - numpy.dot(v, self.b) \
            - numpy.log(1. + numpy.exp(numpy.dot(v, self.W) + self.c)).sum(1)
    
    def gibbs(self, v):
        """
        Perform one Gibbs sampling step.
        
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
    
    def _fit(self, v_pos):
        """
        Adjust the parameters to maximize the likelihood of {\bf v}
        using Stochastic Maximum Likelihood (SML).
        
        Parameters
        ----------
        v_pos: array-like, shape (n_samples, n_visibles)
        """
        h_pos = self.mean_h(v_pos)
        v_neg = self.sample_v(self.h_samples)
        h_neg = self.mean_h(v_neg)
        
        p_pos = v_pos[:, :, None] * h_pos[:, None, :]
        p_neg = v_neg[:, :, None] * h_neg[:, None, :]
        
        self.W += self.epsilon * (p_pos.mean(0) - p_neg.mean(0))
        self.c += self.epsilon * (h_pos.mean(0) - h_neg.mean(0))
        self.b += self.epsilon * (v_pos.mean(0) - v_neg.mean(0))
        
        self.h_samples = numpy.random.binomial(1, h_neg)
    
    def pseudo_likelihood(self, v):
        """
        Compute the pseudo-likelihood of {\bf v}.
        
        Parameters
        ----------
        v: array-like, shape (n_samples, n_visibles)
        
        Returns
        -------
        pseudo_likelihood: array-like, shape (n_samples,)
        """
        fe = self.free_energy(v)
        
        v_ = v.copy()
        i_ = numpy.random.randint(0, v.shape[1], v.shape[0])
        v_[range(v.shape[0]), i_] = v_[range(v.shape[0]), i_] == 0
        fe_ = self.free_energy(v_)
        
        return v.shape[1] * numpy.log(self._sigmoid(fe_ - fe))
    
    def fit(self, X, verbose=False):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        """
        if self.W == None:
            self.W = numpy.random.normal(0, 0.01, (X.shape[1], self.n_hiddens))
            self.c = numpy.zeros(self.n_hiddens)
            self.b = numpy.zeros(X.shape[1])
            self.h_samples = numpy.zeros((self.n_samples, self.n_hiddens))
        
        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = int(numpy.ceil(len(inds) / float(self.n_samples)))
        
        for epoch in range(self.epochs):
            for minibatch in range(n_batches):
                self._fit(X[inds[minibatch::n_batches]])
            
            if verbose:
                pl = self.pseudo_likelihood(X).mean()
            
                print "Epoch %d, Pseudo-Likelihood = %.2f" % (epoch, pl)


def main():
    pass


if __name__ == '__main__':
    main()

