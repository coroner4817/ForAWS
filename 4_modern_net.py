import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist


srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


# ReLU
def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    # input is a 1d vec
    # notice: when calc softmax, every element minus the max to reduce the computation
    # not change the final softmax output
    # dimshuffle is convert a 1d vec to a N*1 array
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.):
    # p is the rate of droping in a certain layer
    # X is each layer
    if p > 0:
        # retain rate
        retain_prob = 1 - p
        # base on the p, generate a X.shape mat with 1 and 0
        # p=0.5 then around half of output is 1
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        # must scale here to perform inverted dropout
        # check cs231 - dropout
        X /= retain_prob
    return X


def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    # drop input layer first
    X = dropout(X, p_drop_input)

    # first hidden layer
    h = rectify(T.dot(X, w_h))
    # drop first hidden layer
    h = dropout(h, p_drop_hidden)

    # sec hidden layer
    h2 = rectify(T.dot(h, w_h2))
    h2 = dropout(h2, p_drop_hidden)

    py_x = softmax(T.dot(h2, w_o))

    return h, h2, py_x


# using this method to dynamic changing the learning rate
# change the learning rate base on the value of the gradients
def RMSprop(cost, params, lr=0.001, rho=0.9, eps=1e-6):
    grads = T.grad(cost=cost, wrt=params)

    updates = []
    for p, g in zip(params, grads):
        # acc is the cache
        # * 0. is for getting the shpe of the p. the cache is init to 0
        acc = theano.shared(p.get_value() * 0.)
        # update the cache, rho is the dacey rate, g is dx
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + eps)
        # before learning rate
        g = g / gradient_scaling
        updates.append((acc, acc_new)) # why need append cache? comment out still works
        updates.append((p, p - lr * g))
    return updates


trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

# input layer dropout 20%, hidden layer dropout 50%, this is for training!!!!
noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
# dropout rate = 0, this is for prediction!!!!
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)

# prediction
y_x = T.argmax(py_x, axis=1)
# for training
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

params = [w_h, w_h2, w_o]

updates = RMSprop(cost, params, lr=0.001)


train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))
