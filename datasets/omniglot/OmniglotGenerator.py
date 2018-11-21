import numpy as np
import keras
from PIL import Image

from datasets.omniglot.OmniglotRawDataset import OmniglotRawDataset


class OmniglotGenerator(keras.utils.Sequence):
    def __init__(self,
                 dataset: OmniglotRawDataset,
                 batch_size: int,
                 epoch_length: int,
                 unique_classes_starting_count: int,
                 max_unique_classes_count: int,
                 samples_per_class: int,
                 image_size,
                 mode="even",
                 ):
        self.dataset = dataset
        self.images_indices_per_label = dataset.get_images_indices_per_label()
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.unique_classes_count = unique_classes_starting_count
        self.max_unique_classes_count = max_unique_classes_count
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.mode = mode

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, _):
        batch_classes = np.random.permutation(np.arange(self.dataset.classes_count))[:self.unique_classes_count]
        batch_images_indices_per_label = self.images_indices_per_label[batch_classes]

        if self.mode == "even":
            labels = np.tile(np.repeat(np.arange(self.unique_classes_count), self.samples_per_class), self.batch_size)
            labels = np.reshape(labels, [self.batch_size, self.sequences_length])
            for i in range(self.batch_size):
                np.random.shuffle(labels[i])
        else:
            batch_labels = np.arange(self.unique_classes_count)
            labels = np.empty([self.batch_size, self.sequences_length], dtype=np.int64)
            for i in range(self.batch_size):
                labels[i] = np.random.choice(batch_labels, self.sequences_length)

        images_indices_in_class = np.random.randint(0, 20, labels.shape)
        batch_indices = batch_images_indices_per_label[labels, images_indices_in_class]
        images = [[self.dataset.images[index] for index in sequence_indices] for sequence_indices in batch_indices]
        images = self.augment_images(images, labels)

        one_hot_labels = keras.utils.to_categorical(labels, self.unique_classes_count)
        offset_one_hot_labels = np.zeros([self.batch_size, 1, self.unique_classes_count], dtype=np.float32)
        offset_one_hot_labels = np.concatenate([offset_one_hot_labels, one_hot_labels[:, :-1]], axis=1)
        inputs = np.concatenate([images, offset_one_hot_labels], axis=-1)

        samples_weights = np.arange(self.sequences_length) / (self.sequences_length - 1)
        samples_weights = np.tile(samples_weights, self.batch_size)
        samples_weights = np.reshape(samples_weights, [self.batch_size, self.sequences_length])
        return inputs, one_hot_labels, samples_weights

    def on_epoch_end(self):
        pass
        # self.unique_classes_count += 1
        # self.unique_classes_count = min(self.unique_classes_count, self.max_unique_classes_count)

    @property
    def sequences_length(self):
        return self.unique_classes_count * self.samples_per_class

    def augment_images(self, images: list, labels: np.ndarray) -> np.ndarray:
        angle_map = np.random.randint(0, 4, [self.batch_size, self.unique_classes_count]) * 90

        pixel_count = np.prod(self.image_size)
        augmented_images = np.empty([self.batch_size, self.sequences_length, pixel_count], dtype=np.float32)
        for batch_index in range(self.batch_size):
            for sample_index in range(self.sequences_length):
                image: Image.Image = images[batch_index][sample_index]
                label: np.int64 = labels[batch_index, sample_index]

                angle = angle_map[batch_index, label] + np.random.rand() * 22.5 - 11.25
                translation = np.random.randint(-10, 11, size=2).tolist()

                image = image.rotate(angle, translate=translation)
                image = image.resize(self.image_size)

                np_image = np.asarray(image)
                np_image = np_image.astype(dtype=np.float32) / 255.0  # normalization: max is 255 all the time (bool)
                np_image = np.reshape(np_image, [-1])
                augmented_images[batch_index, sample_index] = np_image

        return augmented_images
