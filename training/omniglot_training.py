from keras import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import os

from layers.MemoryAugmentedLayer import MemoryAugmentedLayer
from datasets.omniglot.OmniglotRawDataset import OmniglotRawDataset
from datasets.omniglot.OmniglotGenerator import OmniglotGenerator
from training.train_utils import get_log_dir, save_model_info


def train(path,
          train_name="images_background",
          test_name="images_evaluation",
          image_size=(20, 20),
          batch_size=16,
          unique_classes_starting_count=5,
          max_unique_class_count=5,
          samples_per_class=10,
          epochs=100,
          train_epoch_length=1000,
          test_epoch_length=50,
          log_dir="tmp/logs"):
    print("Loading Omniglot...")
    train_set = OmniglotRawDataset(os.path.join(path, train_name))
    test_set = OmniglotRawDataset(os.path.join(path, test_name))

    train_generator = OmniglotGenerator(dataset=train_set,
                                        batch_size=batch_size,
                                        epoch_length=train_epoch_length,
                                        unique_classes_starting_count=unique_classes_starting_count,
                                        max_unique_classes_count=max_unique_class_count,
                                        samples_per_class=samples_per_class,
                                        image_size=image_size)

    test_generator = OmniglotGenerator(dataset=test_set,
                                       batch_size=batch_size,
                                       epoch_length=test_epoch_length,
                                       unique_classes_starting_count=unique_classes_starting_count,
                                       max_unique_classes_count=max_unique_class_count,
                                       samples_per_class=samples_per_class,
                                       image_size=image_size)

    # Note : Use "max_unique_class_count" instead of "unique_classes_starting_count"
    # to increase the number of classes over time (+1 every 10000 episodes in original paper)
    # Maybe sequence_length should be variable (increasing over time, with unique classes count)
    sequence_length = unique_classes_starting_count * samples_per_class
    pixel_count = image_size[0] * image_size[1]

    input_layer = Input(shape=[sequence_length, pixel_count + unique_classes_starting_count])
    layer = input_layer

    mann = MemoryAugmentedLayer(memory_width=40,
                                memory_height=128,
                                usage_decay=0.99,
                                read_heads_count=4,
                                controller_units=200,
                                return_sequences=True)
    mann_initial_state = mann.cell.initial_state(batch_size)
    layer = mann([layer, *mann_initial_state])
    layer = TimeDistributed(Dense(units=unique_classes_starting_count, activation="softmax"))(layer)
    output = layer
    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy",
                  metrics=["acc", acc_at_1st, acc_at_5th, acc_at_10th],
                  sample_weight_mode="temporal")

    log_dir = get_log_dir(log_dir)
    save_model_info(model, log_dir)

    tensorboard_callback = TensorBoard(log_dir=log_dir, batch_size=batch_size, update_freq=2000)

    model.fit_generator(generator=train_generator, epochs=epochs, shuffle=False, validation_data=test_generator,
                        callbacks=[tensorboard_callback])


def acc_at_1st(y_true: tf.Tensor, y_pred: tf.Tensor):
    return accuracy_at_n(y_true, y_pred, 1)


def acc_at_5th(y_true: tf.Tensor, y_pred: tf.Tensor):
    return accuracy_at_n(y_true, y_pred, 5)


def acc_at_10th(y_true: tf.Tensor, y_pred: tf.Tensor):
    return accuracy_at_n(y_true, y_pred, 10)


def accuracy_at_n(y_true: tf.Tensor, y_pred: tf.Tensor, n: int):
    batch_size, sequence_length, unique_class_count = [16, 50, 5]

    y_pred = tf.transpose(tf.argmax(y_pred, axis=-1))
    y_true_one_hot = tf.transpose(y_true, perm=[1, 0, 2])  # (sequence_length, batch_size, unique_class_count)
    y_true = tf.argmax(y_true_one_hot, axis=-1)
    n = tf.constant(n, dtype=tf.int32)  # ()
    accuracy = tf.equal(y_true, y_pred)  # (sequence_length, batch_size)
    accuracy = tf.cast(accuracy, dtype=tf.float32)

    counts = tf.zeros([batch_size, unique_class_count], dtype=tf.int32)
    classes_accuracy = tf.zeros([unique_class_count], dtype=tf.float32)
    for i in range(sequence_length):
        counts = counts + tf.cast(y_true_one_hot[i], dtype=tf.int32)
        at_n = tf.equal(counts, n)  # (batch_size, unique_class_count)
        at_n = tf.cast(at_n, dtype=tf.float32)
        step_accuracy = accuracy[i]  # (batch_size)
        step_accuracy = tf.expand_dims(step_accuracy, axis=-1)  # (batch_size, 1)
        step_accuracy = at_n * step_accuracy * y_true_one_hot[i]  # (batch_size, unique_class_count)
        classes_accuracy = classes_accuracy + tf.reduce_mean(step_accuracy, axis=0)  # (unique_class_count)
    return tf.reduce_mean(classes_accuracy)
