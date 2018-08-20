"""
File contains code for training DNN for cifar10 dataset classification
Trained model is saved in the specified model

Created on: July 17, 2018
Author: Harshal
"""

import tensorflow as tf


if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    BATCH_SIZE = 256
    NO_OF_EPOCHS = 10
    LEARNING_RATE = 10e-5
    LAMBDA = 0.5

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    logits = cifar10_model.dnn(image_queue)
    loss, train_step = cifar10_model.train(logits, label_queue, LEARNING_RATE, LAMBDA, step)
    accuracy = cifar10_model.old_evaluate(logits, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    session_args = {
        'checkpoint_dir': './trained_model',
        'save_checkpoint_steps': 300
    }
    with tf.train.MonitoredTrainingSession(**session_args) as sess:

        count = 1
        for i in range(5):
            cifar10_dataset = cifar10_input.unpickle(filename_list[i])
            image_in = cifar10_dataset[b'data']
            label_in = cifar10_dataset[b'labels']
            sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

            while True:
                try:
                    loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy])
                    if count % 100 == 0:
                        print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = cifar10_dataset[b'data']
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        accuracy_value = sess.run(accuracy)
        print("Accuracy: ", accuracy_value)
