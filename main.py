#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from tensorflow import keras;
import matplotlib.pyplot as plt;

def main():
# load data set
    dataset = keras.datasets.fashion_mnist;
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data();

# classification names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", \
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];

# preprocess dataset such that all image values are in the range of 0 to 1
    train_images = train_images / 255.0;
    test_images = test_images / 255.0;

# build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10)
    ]);

# compile model
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

# train model
    model.fit(train_images, train_labels, epochs=20);

# test model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2);
    print(f"Model accuracy: {test_acc}");

if __name__ == "__main__":
    main();

