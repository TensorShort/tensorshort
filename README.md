![TensorShort Logo](assets/tensorflow_transparent.png)
# TensorShort

Official TensorFlow fork with shorter, cleaner and better naming conventions for functions and methods

Here's a code comparison between an original and TensorShort version of Keras API:

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
# Eager
### Old:
```
tf.enable_eager_execution()
```

### New:
```
eager()
```
