"""
File contains code for training DNN for cifar10 dataset classification
Trained model is saved in the specified model

Created on: July 17, 2018
Author: Harshal
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import os
import shutil

if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 30
    INITIAL_LEARNING_RATE = [10e-4]
    DECAY_STEP = 2000
    DECAY_RATE = 0.1
    LAMBDA = [0.008]

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]
    image_all = np.array([]).reshape((-1, 32, 32, 3))
    for i in range(5):
        cifar10_dataset = cifar10_input.unpickle(filename_list[i])
        image_all = np.append(image_all, cifar10_dataset[b'data'].reshape((-1, 32, 32, 3)), axis = 0)
    mean = image_all.mean(axis = (0, 1, 2)).astype('float32')
    variance = image_all.var(axis = (0, 1, 2)).astype('float32')

    accuracy_value = np.zeros((len(LAMBDA), len(INITIAL_LEARNING_RATE)))

    for index_x, lmbd in enumerate(LAMBDA):
        for index_y, in_lr in enumerate(INITIAL_LEARNING_RATE):
            
            if os.path.exists('./trained_model'):
                shutil.rmtree('./trained_model')
            os.makedirs('./trained_model', exist_ok = True)
            np.savetxt('./trained_model/mean.txt', mean)
            np.savetxt('./trained_model/variance.txt', variance)
            
            tf.reset_default_graph()

            image = tf.placeholder(tf.uint8)
            label = tf.placeholder(tf.int32)
            phase = tf.placeholder(tf.bool)

            dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, 1)
            data = dataset_iterator.get_next()
            image_queue = data["features"]
            label_queue = data["label"]
            
            step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(in_lr, step, DECAY_STEP, DECAY_RATE, staircase = True)
            # learning_rate = tf.constant(in_lr)

            with tf.device('/gpu:0'):
                logits = cifar10_model.dnn(image_queue, mean, variance, phase)
                loss, train_step = cifar10_model.train(logits, label_queue, learning_rate, lmbd, step, tf.trainable_variables())
                accuracy = cifar10_model.old_evaluate(logits, label_queue)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            session_args = {
                'checkpoint_dir': './trained_model',
                'save_checkpoint_steps': 300,
                'config': config
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
                                loss_value, _, lr_value, accuracy_value[index_x, index_y] = sess.run([loss, train_step, learning_rate, accuracy], feed_dict = {phase: True})
                                if count % 20 == 0:
                                    print("Step: %6d,\tEpoch: %4d,\tLearning Rate: %e,\tLoss: %8.4f,\tAccuracy: %0.4f" %
                                          (count, epoch, lr_value, loss_value, accuracy_value[index_x, index_y]))
                                count += 1
                            except tf.errors.OutOfRangeError:
                                break

                cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
                image_in = cifar10_dataset[b'data']
                label_in = cifar10_dataset[b'labels']
                sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

                accuracy_value[index_x, index_y] = sess.run(accuracy, feed_dict = {phase: False})
                print("Accuracy: ", accuracy_value[index_x, index_y])

    print(accuracy_value)
