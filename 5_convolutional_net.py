import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


# ReLU
def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.):
    if p > 0:
        retain_prop = 1 - p
        X *= srng.binomial(X.shape, p=retain_prop, dtype=theano.config.floatX)
        X /= retain_prop
    return X


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scalling = T.sqrt(acc_new + epsilon)
        g = g/gradient_scalling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):

    # conv + ReLU + pool
    # border_mode = full, then zero-padding, default is valid
    l1a = rectify(conv2d(X, w, border_mode='full'))
    # pooling at 2*2 kernel and select the largest in the kernel
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    # conv + ReLU + pool
    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    # conv + ReLU + pool
    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    # convert a ndim array to 2 dim. if l3b dim larger than 2 then the rest dim collapsed.
    # flatten for enter the FC layer
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    # FC + ReLU
    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)


    # output layer + softmax
    pyx = softmax(T.dot(l4, w_o))

    return l1, l2, l3, l4, pyx


trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

# 4 dim tensor. like a 4 dim matrix
X = T.ftensor4()
Y = T.fmatrix()

# conv layer params
# every filter is 3 by 3
# (output channels, input channels, filter rows, filter columns)
# so the first conv layer has 32 filters and has 32*3*3 free params
# input channle is 1 but the elements in the channel is 784
# every neuron's output is 3*3, every channel have a certain number of kernel, depended on the stride
w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))


# FC layer params
# layer 3 output channels is 128
# layer 3 output collapse to 2 dim of (128*3*3) * (num of kernel's width) * (num of kernel's height)
w4 = init_weights((128 * 3 * 3, 625))
# output layer has 625 neurons
w_o = init_weights((625, 10))

# for training
noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5)
# for prediction
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0., 0.)
# final prediction
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)


for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

