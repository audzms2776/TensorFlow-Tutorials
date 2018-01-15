import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Functions and classes for loading and using the Inception model.
import inception

import cifar10
from cifar10 import img_size, num_channels, num_classes
from inception import transfer_values_cache

cifar10.data_path = "/tmp/CIFAR-10/"
cifar10.maybe_download_and_extract()

tf.logging.set_verbosity(tf.logging.INFO)

# Training Parameters
learning_rate = 0.001
num_steps = 4000
batch_size = 200
img_size_cropped = 24

# Network Parameters
dropout = 0.25  # Dropout, probability to drop a unit

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

images_train = images_train[:5000]
cls_train = cls_train[:5000]
labels_train = labels_train[:5000]

images_test = images_test[:100]
cls_test = cls_test[:100]
labels_test = labels_test[:100]

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

inception.data_dir = '/tmp/inception/'
inception.maybe_download()
model = inception.Inception()

file_path_cache_train = os.path.join(cifar10.data_path, 'inception_mini_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_mini_cifar10_test.pkl')

print("Processing Inception transfer-values for training-images ...")

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_train * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = transfer_values_cache(
    cache_path=file_path_cache_train,
    images=images_scaled,
    model=model)

print("Processing Inception transfer-values for test-images ...")

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_test * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(
    cache_path=file_path_cache_test,
    images=images_scaled,
    model=model)


# build network
def cnn_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']

        layer1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
        drop1 = tf.layers.dropout(layer1, dropout, is_training)
        out = tf.layers.dense(drop1, n_classes)

    return out


def model_fn(features, labels, mode):
    # make logits
    logits_train = cnn_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = cnn_net(features, num_classes, dropout, reuse=True, is_training=False)

    # prediction
    prediction = {
        'classes': tf.argmax(logits_test, axis=1)
    }

    # if PREDICT return tf.estimator.EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, prediction['classes'])

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32), name='softmax_tensor'
    ))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        loss=loss_op,
        global_step=tf.train.get_global_step()
    )

    # accuracy
    acc_op = tf.metrics.accuracy(labels, prediction['classes'], name='accuracy')

    return tf.estimator.EstimatorSpec(
        mode,
        prediction['classes'],
        loss_op,
        train_op,
        {'accuracy': acc_op}
    )


def estimator_fn(data, epoch=None, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={'images': data['images']}, y=data['labels'],
        batch_size=batch_size, num_epochs=epoch, shuffle=shuffle
    )


# y 값을 인덱스로 줘야한다.
# array([6, 9, 9, ..., 9, 1, 1])
def train_fn(task, estimator):
    if task == 'train':
        input_fn = estimator_fn({'images': transfer_values_train, 'labels': cls_train})
        estimator.train(input_fn, steps=num_steps)
    elif task == 'test':
        input_fn = estimator_fn({'images': transfer_values_test, 'labels': cls_test},
                                epoch=1, shuffle=False)
        estimator.evaluate(input_fn)


def main(_):
    cnn_estimator = tf.estimator.Estimator(model_fn, model_dir='/tmp/cifar-model5')

    tf.train.LoggingTensorHook(tensors={
        'probalities': 'softmax_tensor',
        'good': 'accuracy'
    }, every_n_iter=50)

    train_fn('train', cnn_estimator)
    train_fn('test', cnn_estimator)


if __name__ == '__main__':
    tf.app.run()
