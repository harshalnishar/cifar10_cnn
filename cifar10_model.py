"""
File contains neural network model for cifar10 dataset classification

Created on: July 13, 2018
Author: Harshal
"""

import tensorflow as tf

def dnn(image):
    """
    function defining cifar10 model
    :param image: input image tensor
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 32, 32, 3])

    conv1 = tf.layers.conv2d(image_reshape, 32, [3, 3], strides = (1, 1), padding = "same",
                             activation = tf.nn.relu, name = 'conv2d_1')
    pool1 = tf.layers.max_pooling2d(conv1, pool_size = [2, 2], strides = 2)

    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], strides = (1, 1), padding = "same",
                             activation = tf.nn.relu, name = 'conv2d_2')
    pool2 = tf.layers.max_pooling2d(conv2, pool_size = [2, 2], strides = 2)

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    fc1 = tf.layers.dense(pool2_flat, units = 10, name = 'dense_1')

    return fc1

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

def train(logits, labels, learning_rate):
    """
    function to train the dnn model for cifar10 training set
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :param learning_rate: initial learning rate
    :return: training loss and training operation
    """
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer_step = optimizer.minimize(loss)
    return loss, optimizer_step


if __name__ == "__main__":
    import cifar10_input

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 1000
    LEARNING_RATE = 10e-5

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    logits = dnn(image_queue)
    loss, train_step = train(logits, label_queue, LEARNING_RATE)
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

        cifar10_dataset = cifar10_input.unpickle(filename_list[1])
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        accuracy_value = sess.run(accuracy)
        print("Accuracy: ", accuracy_value)
