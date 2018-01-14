import tensorflow as tf
import cifar10
from cifar10 import img_size, num_channels, num_classes

cifar10.data_path = "/tmp/CIFAR-10/"
cifar10.maybe_download_and_extract()

tf.logging.set_verbosity(tf.logging.INFO)

# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 200
img_size_cropped = 24

# Network Parameters
dropout = 0.25  # Dropout, probability to drop a unit

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


# build network
def cnn_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, img_size, img_size, num_channels])

        conv1 = tf.layers.conv2d(x, 20, 5, activation=tf.nn.relu)
        batch_norm1 = tf.layers.batch_normalization(conv1, training=is_training)
        bn_acti1 = tf.nn.relu(batch_norm1)
        pool1 = tf.layers.max_pooling2d(bn_acti1, 2, 2)

        conv2 = tf.layers.conv2d(pool1, 30, 3, activation=tf.nn.relu)
        batch_norm2 = tf.layers.batch_normalization(conv2, training=is_training)
        bn_acti2 = tf.nn.relu(batch_norm2)
        pool2 = tf.layers.max_pooling2d(bn_acti2, 2, 2)

        fc1 = tf.contrib.layers.flatten(pool2)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

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
def train_fn(task, model):
    if task == 'train':

        input_fn = estimator_fn({'images': images_train, 'labels': cls_train})
        model.train(input_fn, steps=num_steps)
    elif task == 'test':
        input_fn = estimator_fn({'images': images_test, 'labels': cls_test},
                                epoch=1, shuffle=False)
        model.evaluate(input_fn)


def visual_model(model):
    n_images = 4
    # Get images from test set
    test_images = images_test[:n_images]
    test_labels = labels_test[:n_images]

    # Prepare the input data
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_images}, shuffle=False)
    # Use the model to predict the images class
    preds = list(model.predict(input_fn))

    print(test_labels)
    print(preds)


def main(_):
    model = tf.estimator.Estimator(model_fn, model_dir='/tmp/cifar-model3')

    tf.train.LoggingTensorHook(tensors={
        'probalities': 'softmax_tensor',
        'good': 'accuracy'
    }, every_n_iter=50)

    train_fn('train', model)
    train_fn('test', model)

    # visual_model(model)


if __name__ == '__main__':
    tf.app.run()
