#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x_sum = np.sum(x)
    x = x / x_sum
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    #     print("+++++++++++++++++++++ softmaxCostAndGradient +++++++++++++++++++++++")
    #     print("The shape of predicted(v_c) is {}, which means each word is presented by {} dims.".format(predicted.shape, predicted.shape[0]))
    #     print("target(o)'s type is {}, and it's value is {},the u_o now is u_target.".format(type(target), target))
    #     print("The shape of outputVectors(u_w) is {}, which means we have {} words.".format(outputVectors.shape, outputVectors.shape[0]))
    y_hat = softmax(np.matmul(outputVectors, predicted))
    #     print("y_hat is{}.".format(y_hat))
    #     print("Then we should minus 1 at the location at {}".format(target+1))
    cost = -np.log(y_hat[target])
    y_hat[target] = y_hat[target] - 1
    dy = y_hat.copy()

    #     print("so we can get the dy:{}".format(y_hat))
    #     print("To get the gradPred, according to the wirte solution what we should know the shapes of dy{} and outputVectors{}".
    #           format(dy.shape, outputVectors.shape))
    gradPred = np.matmul(dy.T, outputVectors)
    #     print("we can get the gradPred easily in shape{}".format(gradPred.shape))
    #     print("To get the grad, according to the wirte solution what we should know the shapes of dy{} and predicted{}".
    #           format(dy.shape, predicted.shape))
    grad = np.outer(dy, predicted)
    #     print("we can get the grad easily in shape{}".format(grad.shape))
    #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    u_o, v_c = outputVectors[target], outputVectors[indices[1:]]
    loss = -np.log(sigmoid(np.matmul(u_o, predicted)))
    print(u_o)
    print(v_c)
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)  #
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    #     print("+++++++++++++++++++++       skipgram       +++++++++++++++++++++++")
    #     print("the shape of inputVectors{}".format(inputVectors.shape))
    #     print("the shape of outputVectors{}".format(outputVectors.shape))
    #     print("Is the inputVectors the same as outputVectors? {}".format(inputVectors==outputVectors))
    # the inputVectors and the outputVectors can be regrad as separate word_embeddings

    # skipgram is a model giving one center word and predict 2*C word around it
    v_c_index = tokens[currentWord]  # the index of center word
    v_c = inputVectors[v_c_index]  # the vectors presenting the center word
    for i in contextWords:  # caculate some parameters between 2*C words and the center word
        w_j_index = tokens[i]  # one in 2*C
        # negSamplingCostAndGradient(predicted, target, outputVectors, dataset,K=10):
        cost_j, gradIn_j, gradOut_j = word2vecCostAndGradient(v_c, w_j_index, outputVectors, dataset)
        cost += cost_j
        gradOut += gradOut_j
        gradIn[v_c_index] += gradIn_j
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    w_c = outputVectors[tokes[currentWords]]
    v_sum = np.sum(inputVectors[tokens[contextWords]], axis=0)
    cost, gradIn, gradOut = word2vecCostAndGradient(w_c, v_sum, outputVectors, dataset)
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N // 2, :]
    outputVectors = wordVectors[N // 2:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N // 2, :] += gin / batchsize / denom
        grad[N // 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient))


test_word2vec()