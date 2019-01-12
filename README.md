![TensorShort Logo](assets/tensorflow_transparent.png)
# TensorShort

Official TensorFlow fork with shorter, cleaner and better naming conventions for functions and methods

1. Use ```t.``` for TensorFlow and ```k.``` for Keras API before defining methods
2. See [full changelog here](./changelog.md)

Here are some code comparisons between an original and TensorShort versions:

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
# CNN layer with Estimators
### Old:
```
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

```
### New:
```
  pool2_flat = t.reshape(pool2, [-1, 7 * 7 * 64])
  dense = t.dense(input=pool2_flat, units=1024, activation=t.relu)
  dropout = t.dropout(
      input=dense, rate=0.4, train=mode == estimator.TRAIN)
```
