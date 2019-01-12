![TensorShort Logo](assets/tensorflow_transparent.png)
# TensorShort

Official TensorFlow fork with shorter, cleaner and better naming conventions for functions and methods

Here's a code comparison between an original and TensorShort version:

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
model = sequential([
    flatten(input_shape=(28, 28)),
    dense(128, activation=relu),
    dense(10, activation=softmax)
])
```
