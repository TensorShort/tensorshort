![TensorShort Logo](assets/tensorflow_transparent.png)
# TensorShort

Official TensorFlow fork with shorter, cleaner and better naming conventions for functions and methods

Here's a code comparison between an original and TensorShort version of Keras API:

# Keras API
### Old:

```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

### New:

```
model = k.sequential([
    k.flatten(input_shape=(28, 28)),
    k.dense(128, activation=relu),
    k.dense(10, activation=softmax)
])
```
# Eager execution
### Old:
```
tf.enable_eager_execution()
```

### New:
```
eager()
```
# CNN with Estimators
### Old:
```
def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

```
### New:
```
def cnn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.conv2d(
      input=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.relu)

  pool1 = tf.max_pooling2d(input=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.conv2d(
      input=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.relu)
  pool2 = tf.max_pooling2d(input=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.dense(input=pool2_flat, units=1024, activation=tf.relu)
  dropout = tf.dropout(
      input=dense, rate=0.4, train=mode == tf.estimator.ModeKeys.TRAIN)
```
