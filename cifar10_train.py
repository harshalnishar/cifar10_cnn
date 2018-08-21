"""
File contains code for training DNN for cifar10 dataset classification
Trained model is saved in the specified model

Created on: July 17, 2018
Author: Harshal
"""

import tensorflow as tf
from random import shuffle

if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 100
    INITIAL_LEARNING_RATE = 10e-5
    DECAY_STEP = 10000
    DECAY_RATE = 0.1
    LAMBDA = 0.01

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, 1)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, step, DECAY_STEP, DECAY_RATE, staircase = True)
    logits = cifar10_model.dnn(image_queue)
    loss, train_step = cifar10_model.train(logits, label_queue, learning_rate, LAMBDA, step)
    accuracy = cifar10_model.old_evaluate(logits, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    session_args = {
        'checkpoint_dir': './trained_model',
        'save_checkpoint_steps': 300
    }
    with tf.train.MonitoredTrainingSession(**session_args) as sess:

        count = 1
        for epoch in range(NO_OF_EPOCHS):
            l = list(range(5))
            shuffle(l)
            for i in l:
                cifar10_dataset = cifar10_input.unpickle(filename_list[i])
                image_in = cifar10_dataset[b'data']
                label_in = cifar10_dataset[b'labels']
                sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

                while True:
                    try:
                        loss_value, _, lr_value, accuracy_value = sess.run([loss, train_step, learning_rate, accuracy])
                        if count % 100 == 0:
                            print("Step: %6d,\tEpoch: %4d,\tLearning Rate: %e,\tLoss: %8.4f,\tAccuracy: %0.4f" %
                                  (count, epoch, lr_value, loss_value, accuracy_value))
                        count += 1
                    except tf.errors.OutOfRangeError:
                        break

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        accuracy_value = sess.run(accuracy)
        print("Accuracy: ", accuracy_value)
