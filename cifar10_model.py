"""
File contains neural network model for cifar10 dataset classification

Created on: July 13, 2018
Author: Harshal
"""

import tensorflow as tf

def dnn(image, mean, variance, phase):
    """
    function defining cifar10 model
    :param image: input image tensor in the flattened form
    :return: model output tensor node
    """

    image = tf.cast(image, tf.float32)
    image_reshape = tf.reshape(image, [-1, 32, 32, 3])

    image_norm = tf.nn.batch_normalization(image_reshape, mean, variance, None, None, 0.0001)


    conv1 = tf.layers.conv2d(image_norm, 32, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_1')
    #mean1, variance1 = tf.nn.moments(pool1, axes = [0, 1, 2])
    conv1_norm = tf.layers.batch_normalization(
        conv1,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu1 = tf.nn.relu(conv1_norm)
    #relu1 = tf.nn.relu(conv1)
    
    conv1_1 = tf.layers.conv2d(relu1, 32, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_1_1')
    #mean1, variance1 = tf.nn.moments(pool1, axes = [0, 1, 2])
    conv1_1_norm = tf.layers.batch_normalization(
        conv1_1,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu1_1 = tf.nn.relu(conv1_1_norm)

    pool1 = tf.layers.max_pooling2d(relu1_1, pool_size = [2, 2], strides = 2)


    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_2')
    conv2_norm = tf.layers.batch_normalization(
        conv2,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu2 = tf.nn.relu(conv2_norm)
    #relu2 = tf.nn.relu(conv2)

    conv2_1 = tf.layers.conv2d(relu2, 64, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_2_1')
    conv2_1_norm = tf.layers.batch_normalization(
        conv2_1,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu2_1 = tf.nn.relu(conv2_1_norm)
    
    pool2 = tf.layers.max_pooling2d(relu2_1, pool_size = [2, 2], strides = 2)


    conv3 = tf.layers.conv2d(pool2, 128, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_3')
    conv3_norm = tf.layers.batch_normalization(
        conv3,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu3 = tf.nn.relu(conv3_norm)
    #relu3 = tf.nn.relu(conv3)
    
    conv3_1 = tf.layers.conv2d(relu3, 128, [3, 3], strides = (1, 1), padding = "same",
                             activation = None, name = 'conv2d_3_1')
    conv3_1_norm = tf.layers.batch_normalization(
        conv3_1,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    relu3_1 = tf.nn.relu(conv3_1_norm)

    pool3 = tf.layers.max_pooling2d(relu3_1, pool_size = [2, 2], strides = 2)


    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])
    fc1 = tf.layers.dense(pool3_flat, units = 256, name = 'dense_1')
    fc1_norm = tf.layers.batch_normalization(
        fc1,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    fc1_relu = tf.nn.relu(fc1_norm)
    #fc1_relu = tf.nn.relu(fc1)

    fc2 = tf.layers.dense(fc1_relu, units = 128, name = 'dense_2')
    fc2_norm = tf.layers.batch_normalization(
        fc2,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    fc2_relu = tf.nn.relu(fc2_norm)
    #fc1_relu = tf.nn.relu(fc1)
    
    fc3 = tf.layers.dense(fc2_relu, 10, name = 'dense_3')
    fc3_norm = tf.layers.batch_normalization(
        fc3,
        #axis = 3,
        momentum = 0.99,
        epsilon = 0.001,
        center = True,
        scale = True,
        training = phase,
        trainable = True
    )
    fc3_softmax = tf.nn.softmax(fc3_norm)
    #fc2_softmax = tf.nn.softmax(fc2)
    
    return fc3_softmax

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
    tf.cast(loss, dtype = tf.float64)
    cost = tf.identity(loss)
    if l2_regularization is not None:
        loss_l2 = tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables() if (('bias' not in v.name) and ('batch_normalization' not in v.name))])
        cost = cost + l2_regularization * loss_l2
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_step = optimizer.minimize(cost, global_step = step, var_list = train_var)
    return loss, optimizer_step


if __name__ == "__main__":
    import cifar10_input

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 1
    LEARNING_RATE = 10e-2
    LAMBDA = None #0.05

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)
    phase = tf.placeholder(tf.bool)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    logits = dnn(image_queue, 0.0, 1.0, phase)
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
                loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy], feed_dict = {phase: True})
                if count % 100 == 0:
                    print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                    print(sess.run(tf.get_default_graph().get_tensor_by_name('batch_normalization/moving_mean:0')))
                count += 1
            except tf.errors.OutOfRangeError:
                break

        variables = [v.name for v in tf.trainable_variables()]
        print(variables)
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        accuracy_value = sess.run(accuracy, feed_dict = {phase: False})
        print("Accuracy: ", accuracy_value)
        print((tf.get_default_graph().get_tensor_by_name('conv2d_1/kernel:0')))
