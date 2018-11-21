import os
from PIL import Image
import numpy as np
from typing import Tuple


class OmniglotRawDataset(object):
    def __init__(self, dataset_path, image_size: Tuple[int, int] = None):
        self.dataset_path = dataset_path
        self.alphabets = dirs_in(dataset_path, True)
        self.alphabets_dirs = dirs_in(dataset_path, True)
        self.characters_dirs = flatten_list([dirs_in(alphabet, True) for alphabet in self.alphabets_dirs])
        self.classes_count = len(self.characters_dirs)
        self.image_size = [105, 105] if image_size is None else image_size

        self.images = []
        for i, character_dir in enumerate(self.characters_dirs):
            characters = files_in(character_dir, True)
            for character_path in characters:
                image = Image.open(character_path)
                image = image.convert(mode="L")
                if image_size is not None:
                    image = image.resize(image_size)
                self.images.append(image)

        self._np_images: np.ndarray = None
        self.labels = np.repeat(np.arange(self.classes_count, dtype=np.int64), 20)

    @property
    def np_images(self):
        if self._np_images is None:
            self._np_images = np.empty([len(self.images), *self.image_size], dtype=np.float32)
            for i in range(len(self.images)):
                image = self.images[0]
                image = np.asarray(image)
                image = image.astype(dtype=np.float32) / 255.0
                self._np_images[i] = image

        return self._np_images

    def shuffle(self):
        permutations = np.random.permutation(np.arange(self.images.shape[0]))
        self.images = self.images[permutations]
        self.labels = self.labels[permutations]

    def get_images_indices_per_label(self):
        images_indices_per_label = np.empty([self.classes_count, 20], dtype=np.int64)
        for i in range(self.classes_count):
            indices = np.argwhere(self.labels == i)
            images_indices_per_label[i] = np.squeeze(indices)
        return images_indices_per_label


def dirs_in(folder: str, full_path: bool):
    return entries_in(folder, full_path, os.path.isdir)


def files_in(folder: str, full_path: bool):
    return entries_in(folder, full_path, os.path.isfile)


def entries_in(folder: str, full_path: bool, condition_fn):
    entries = os.listdir(folder)
    if full_path:
        result = []
        for entry in entries:
            path = os.path.join(folder, entry)
            if condition_fn(path):
                result.append(path)
        return result
    else:
        return [entry for entry in entries if condition_fn(os.path.join(folder, entry))]


def flatten_list(nested_list: list):
    return [item for sublist in nested_list for item in sublist]
