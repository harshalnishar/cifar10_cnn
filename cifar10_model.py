"""
File contains neural network model for cifar10 dataset classification

Created on: July 13, 2018
Author: Harshal
"""

import tensorflow as tf

def dnn(image, mean, variance):
    """
    function defining cifar10 model
    :param image: input image tensor
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 32, 32, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)

    conv1 = tf.layers.conv2d(image_norm, 32, [3, 3], strides = (1, 1), padding = "same",
                             activation = tf.nn.relu, name = 'conv2d_1')
    pool1 = tf.layers.max_pooling2d(conv1, pool_size = [2, 2], strides = 2)

    mean1, variance1 = tf.nn.moments(pool1, axes = [0, 1, 2])
    pool1_norm = tf.nn.batch_normalization(pool1, mean1, variance1, None, None, 0.0001)

    conv2 = tf.layers.conv2d(pool1_norm, 64, [3, 3], strides = (1, 1), padding = "same",
                             activation = tf.nn.relu, name = 'conv2d_2')
    pool2 = tf.layers.max_pooling2d(conv2, pool_size = [2, 2], strides = 2)

    mean2, variance2 = tf.nn.moments(pool2, axes = [0, 1, 2])
    pool2_norm = tf.nn.batch_normalization(pool2, mean2, variance2, None, None, 0.0001)

    conv3 = tf.layers.conv2d(pool2_norm, 128, [3, 3], strides = (1, 1), padding = "same",
                             activation = tf.nn.relu, name = 'conv2d_3')
    pool3 = tf.layers.max_pooling2d(conv3, pool_size = [2, 2], strides = 2)

    mean3, variance3 = tf.nn.moments(pool3, axes = [0, 1, 2])
    pool3_norm = tf.nn.batch_normalization(pool3, mean3, variance3, None, None, 0.0001)

    pool3_flat = tf.reshape(pool3_norm, [-1, 4 * 4 * 128])
    fc1 = tf.layers.dense(pool3_flat, units = 64, name = 'dense_1')

    mean3, variance3 = tf.nn.moments(fc1, axes = [0])
    fc1_norm = tf.nn.batch_normalization(fc1, mean3, variance3, None, None, 0.0001)

    fc2 = tf.layers.dense(fc1_norm, 10, name = 'dense_2')

    return fc2

def predict(logits):
    """
    function outputs the predicted class based on input logits
    :param logits: logits tensor
    :return: returns predicted output with prediction probability
    """
    prediction = tf.cast(tf.argmax(logits, axis = 1), tf.int32)
    probability = tf.reduce_max(tf.nn.softmax(logits), axis = 1)
    return prediction, probability

def old_evaluate(logits, labels):
    """
    function to evaluate the logits output against labels with normal accuracy
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :return: prediction accuracy
    """
    prediction, _ = predict(logits)
    match = tf.equal(labels, prediction)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
    return accuracy

def evaluate(logits, labels):
    """
    function to evaluate the logits output against labels with running sum accuracy
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :return: prediction running accuracy
    """
    prediction, _ = predict(logits)
    accuracy, accuracy_op = tf.metrics.accuracy(labels = labels, predictions = prediction)
    return accuracy, accuracy_op

def train(logits, labels, learning_rate, l2_regularization, step, train_var):
    """
    function to train the dnn model for cifar10 training set
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :param learning_rate: initial learning rate or learning rate function
    :param l2_regularization: regularization factor
    :param step: global training step needed for learning rate function
    :param train_var: list of tensor variables which must be used for minimization
    :return: training loss and training operation
    """
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    cost = tf.identity(loss)
    tf.cast(loss, dtype = tf.float64)
    if l2_regularization is not None:
        loss_l2 = tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        cost = cost + l2_regularization * loss_l2
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    optimizer_step = optimizer.minimize(cost, global_step = step, var_list = train_var)
    return loss, optimizer_step


if __name__ == "__main__":
    import cifar10_input

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 1000
    LEARNING_RATE = 10e-5
    LAMBDA = 0.5

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    logits = dnn(image_queue, 0.0, 1.0)
    loss, train_step = train(logits, label_queue, LEARNING_RATE, LAMBDA, step, tf.trainable_variables())
    accuracy = old_evaluate(logits, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cifar10_dataset = cifar10_input.unpickle(filename_list[0])
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        count = 1
        while True:
            try:
                loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy])
                if count % 100 == 0:
                    print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                count += 1
            except tf.errors.OutOfRangeError:
                break

        variables = [v.name for v in tf.trainable_variables()]
        print(variables)

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        accuracy_value = sess.run(accuracy)
        print("Accuracy: ", accuracy_value)
