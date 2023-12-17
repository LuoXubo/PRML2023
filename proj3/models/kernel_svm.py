from __future__ import print_function
import numpy as np
from random import shuffle
from sklearn.svm import SVC
from .softmax import *

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W * 2
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  score = X.dot(W)
  correct_class_score = score[range(X.shape[0]), y]
  correct_class_score = correct_class_score.reshape(X.shape[0], -1)
  margin = score - correct_class_score + 1
  margin = np.maximum(margin, 0)
  margin[range(X.shape[0]), y] = 0
  loss = np.sum(margin)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  
  ########################Sum#####################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin > 0] = 1
  rowSum = np.sum(margin, axis=1)
  margin[range(margin.shape[0]), y] = -rowSum
  dW = np.dot(X.T, margin) / X.shape[0]
  dW += reg * W * 2

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

class KernelSVMClassifier(object):

  def __init__(self):
    self.W = None
    self.svm = SVC(kernel='rbf', C=1, gamma='scale')

  def train(self, X, y, X_test, y_test, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    
    num_train, dim = X.shape
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      idxs = np.random.choice(num_train, batch_size)
      X_batch = X[idxs, :]
      y_batch = y[idxs]

      # evaluate loss and gradient
      self.svm.fit(X_batch, y_batch)
      y_pred = self.svm.predict(X_test)
      loss = np.mean(y_pred != y_test)
    #   loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))
  
  def predict(self, X):
    y_pred = self.svm.predict(X)
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class KernelSVM(KernelSVMClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(KernelSVMClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

