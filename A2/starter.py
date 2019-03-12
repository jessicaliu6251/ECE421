import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""Define Macros"""
dropout = True
reg_on = False
reg = 0.5
p = 0.9

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


# data flatten
def flatten():
    trainData_flatten = np.zeros((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    for i in range(trainData.shape[0]):
        trainData_flatten[i] = np.ndarray.flatten(trainData[i])
    valData_flatten = np.zeros((validData.shape[0], validData.shape[1]*validData.shape[2]))
    for i in range(validData.shape[0]):
        valData_flatten[i] = np.ndarray.flatten(validData[i])
    testData_flatten = np.zeros((testData.shape[0], testData.shape[1]*testData.shape[2]))
    for i in range(testData.shape[0]):
        testData_flatten[i] = np.ndarray.flatten(testData[i])

    return trainData_flatten, valData_flatten, testData_flatten


"""
def relu(x):
    # TODO

def softmax(x):
    # TODO


def computeLayer(X, W, b):
    # TODO

def CE(target, prediction):

    # TODO

def gradCE(target, prediction):

    # TODO
"""

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

train_data = trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])
valid_data = validData.reshape(validData.shape[0], validData.shape[1]*validData.shape[2])
test_data = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2])

train_target, valid_target, test_target = convertOneHot(trainTarget, validTarget, testTarget)
trainData_flatten, valData_flatten, testData_flatten = flatten()


def accuracy(y_hat, y):
    count = 0
    for i in range(len(y)):
        if y_hat[i] == y[i]:
            count += 1
    acc = count/len(y)
    return acc


# plot part2
def plot_loss_acc_part2(epochs, loss_train, loss_val, loss_test, acc_train, acc_val, acc_test, lossType=None):
    plt.figure(num=1, figsize=(20, 10))
    plt.subplot(121)
    plt.plot(epochs, loss_train, 'red', label='Training loss')
    plt.plot(epochs, loss_val, 'green', label='Validation loss')
    plt.plot(epochs, loss_test, 'yellow', label='Test loss')
    if dropout:
        title = 'SGD, Type={lossType}, Loss\nreg = 0.00, p = {keep_prob}'.format(lossType=lossType, keep_prob=p)
    elif reg_on:
        title = 'SGD, Type={lossType}, Loss\nreg = {reg}, no dropout'.format(lossType=lossType, reg=reg)
    else:
        title = 'SGD, Type={lossType}, Loss\nreg = 0.00, no dropout'.format(lossType=lossType)

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val', 'Test'], loc='upper right')

    plt.subplot(122)
    plt.plot(epochs, acc_train, 'red', label='Training accuracy')
    plt.plot(epochs, acc_val, 'green', label='Validation accuracy')
    plt.plot(epochs, acc_test, 'yellow', label='Test accuracy')
    if dropout:
        title = 'SGD, Type={lossType}, Accuracy\nreg = 0.00, p = {keep_prob}'.format(lossType=lossType, keep_prob=p)
    elif reg_on:
        title = 'SGD, Type={lossType}, Accuracy\nreg = {reg}, no dropout'.format(lossType=lossType, reg=reg)
    else:
        title = 'SGD, Type={lossType}, Accuracy\nreg = 0.00, no dropout'.format(lossType=lossType)

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val', 'Test'], loc='lower right')
    if dropout:
        plt.savefig("part2_{lossType}_reg=0.00_p={keep_prob}.png".format(lossType=lossType, keep_prob=p))
    elif reg_on:
        plt.savefig("part2_{lossType}_reg={reg}_no_dropout.png".format(reg=reg, lossType=lossType))
    else:
        plt.savefig("part2_{lossType}_reg=0.00_no_dropout.png".format(lossType=lossType))

    plt.show()


def build_nn_tf(num_input_channels, num_filters, filter_shape, pool_shape, learning_rate):

    tf.set_random_seed(421)

    X = tf.placeholder(tf.float32, [None, 784], name="inputs")
    X_shaped = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.int32, [None, 10], name="labels")
    mode = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter with Xavier scheme
    xavier_initializer = tf.contrib.layers.xavier_initializer()
    # regularizer to add penalty to large weights
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg)
    filters = tf.get_variable(name="conv_filterss", shape=conv_filter_shape, initializer=xavier_initializer,
                              regularizer=regularizer)
    bias = tf.get_variable(name="conv_bias", initializer=tf.truncated_normal(shape=[num_filters]))

    # setup the convolutional layer operation
    conv_layer = tf.nn.conv2d(X_shaped, filters, strides=[1, 2, 2, 1], padding='SAME')

    # add the bias
    conv_layer += bias

    # apply a ReLU non-linear activation
    relu_layer = tf.nn.relu(conv_layer)

    # batch normalization
    mean, variance = tf.nn.moments(relu_layer, [0, 1, 2])
    batch_norm_layer = tf.nn.batch_normalization(relu_layer, mean, variance, None, None, 1e-4)

    # now perform max pooling of size 2x2
    pooling_ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 1, 1]
    max_pooling_layer = tf.nn.max_pool(batch_norm_layer, ksize=pooling_ksize, strides=strides, padding='SAME')

    # flatten the output from conv_layer, ***** need to check flattened shape *****
    flattened = tf.reshape(max_pooling_layer, [-1, 14 * 14 * num_filters])

    # setup weights and bias values for the first fully connected layer, then activate with ReLU
    # need to check shape for weights and biases
    wd1 = tf.get_variable(initializer=xavier_initializer, shape=[14 * 14 * num_filters, 784], name='wd1',
                          regularizer=regularizer)
    bd1 = tf.get_variable(initializer=tf.truncated_normal(shape=[784]), name='bd1')
    fully_connected1 = tf.matmul(flattened, wd1) + bd1

    # DROP-OUT here
    if dropout:
        drop_out = tf.nn.dropout(fully_connected1, keep_prob)
        fully_connected1 = tf.nn.relu(drop_out)
    else:
        fully_connected1 = tf.nn.relu(fully_connected1)

    # setup weights and bias values for the second fully connected and softmax layer
    wd2 = tf.get_variable(initializer=xavier_initializer, shape=[784, 10], name='wd2', regularizer=regularizer)
    bd2 = tf.get_variable(initializer=tf.truncated_normal(shape=[10]), name='bd2')
    fully_connected2 = tf.matmul(fully_connected1, wd2) + bd2

    # compute predictions and loss
    y_pred = tf.argmax(input=fully_connected2, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected2, labels=Y))

    if reg_on:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        loss += reg_term

    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = adam.minimize(loss=loss)
    return optimizer, loss, y_pred, X, Y, mode, keep_prob


def sgd_tensorflow(mini_batch_size, learning_rate, training_epochs, num_input_channels, num_filters, filter_shape):

    optimizer, loss, y_pred, X, Y, mode, keep_prob = build_nn_tf(num_input_channels=num_input_channels,
                                                                 num_filters=num_filters, filter_shape=filter_shape,
                                                                 pool_shape=[2, 2], learning_rate=learning_rate)

    loss_train = np.zeros(training_epochs)
    loss_val = np.zeros(training_epochs)
    loss_test = np.zeros(training_epochs)
    acc_train = np.zeros(training_epochs)
    acc_val = np.zeros(training_epochs)
    acc_test = np.zeros(training_epochs)

    init = tf.global_variables_initializer()
    is_training = True
    not_training = False

    # Trying to get GPU working
    """ 
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    """

    # change labels to [len, 10], so they can be passed to the CNN implemented using tensorflow
    labels_train_all = np.zeros([len(trainTarget), 10])
    labels_valid = np.zeros([len(validTarget), 10])
    labels_test = np.zeros([len(testTarget), 10])
    for i in range(len(trainTarget)):
        labels_train_all[i, trainTarget[i]] = 1
    for i in range(len(validTarget)):
        labels_valid[i, validTarget[i]] = 1
    for i in range(len(testTarget)):
        labels_test[i, testTarget[i]] = 1

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            # shuffling data for each epoch
            trainData_shuffled, trainTarget_shuffled = shuffle(trainData_flatten, trainTarget)
            for num_batch in range(int(len(trainTarget) / mini_batch_size)):
                labels_shuffled = trainTarget_shuffled[num_batch * mini_batch_size:(num_batch + 1) * mini_batch_size]
                labels_train = np.zeros([mini_batch_size, 10])

                for i in range(mini_batch_size):
                    labels_train[i, labels_shuffled[i]] = 1

                _ = sess.run([optimizer], feed_dict={
                    X: trainData_shuffled[num_batch * mini_batch_size:(num_batch + 1) * mini_batch_size],
                    Y: labels_train, mode: is_training, keep_prob: p})

            # obtaining loss/accuracy values on training, validation and test data for each epoch
            loss_train_per_epoch, y_train_hat = sess.run([loss, y_pred], feed_dict={
                X: trainData_flatten, Y: labels_train_all, mode: not_training, keep_prob: p})
            loss_val_per_epoch, y_valid_hat = sess.run([loss, y_pred], feed_dict={
                X: valData_flatten, Y: labels_valid, mode: not_training, keep_prob: p})
            loss_test_per_epoch, y_test_hat = sess.run([loss, y_pred], feed_dict={
                X: testData_flatten, Y: labels_test, mode: not_training, keep_prob: p})
            loss_train[epoch] = loss_train_per_epoch
            loss_val[epoch] = loss_val_per_epoch
            loss_test[epoch] = loss_test_per_epoch
            acc_train[epoch] = accuracy(y_train_hat, trainTarget)
            acc_val[epoch] = accuracy(y_valid_hat, validTarget)
            acc_test[epoch] = accuracy(y_test_hat, testTarget)
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "%.9f" % loss_train_per_epoch)
            print("Epoch:", '%04d' % (epoch + 1), "acc=", "%.9f" % acc_test[epoch])

    # plot curves
    epochs = range(1, training_epochs + 1)
    plot_loss_acc_part2(epochs, loss_train, loss_val, loss_test, acc_train, acc_val, acc_test, lossType="CE")
    print("loss_train = {Loss_train}\n"
          "loss_val = {Loss_val}\n"
          "loss_test = {Loss_test}\n"
          "acc_train = {Acc_train}\n"
          "acc_val = {Acc_val}\n"
          "acc_test = {Acc_test}\n".format(Loss_train=loss_train[training_epochs - 1],
                                           Loss_val=loss_val[training_epochs - 1],
                                           Loss_test=loss_test[training_epochs - 1],
                                           Acc_train=acc_train[training_epochs - 1],
                                           Acc_val=acc_val[training_epochs - 1],
                                           Acc_test=acc_test[training_epochs - 1])
          )
    return


sgd_tensorflow(mini_batch_size=32, learning_rate=1e-4, training_epochs=50, num_input_channels=1,
               num_filters=32, filter_shape=[3, 3])
