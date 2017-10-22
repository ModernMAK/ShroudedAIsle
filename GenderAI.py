import numpy as np
import os
from os import path
from PIL import Image


def get_image_and_label_from_image(file_path):
    gender = get_gender_from_image(file_path)
    image = get_image_from_file(file_path)
    gender_one_hot = get_one_hot(gender, 2)
    return image, gender_one_hot


def get_image_and_label_from_image_and_reshape(file_path, shape, channels):
    image, gender = get_image_and_label_from_image(file_path)
    image = reshape_image(image, shape, channels)
    return image, gender


def get_gender_from_image(file_path):
    folder_path = path.dirname(file_path)
    folder = path.basename(folder_path)

    male = ["Male", "male", "Men", "men", "M", "m"]
    female = ["Female", "female", "Women", "Women", "W", "w"]

    if folder in male:
        return 0
    if folder in female:
        return 1
    raise Exception("(%s) folder name invalid! From (%s)" % (folder, folder_path))


# Returns a numpy array
def get_image_from_file(file):
    pil_img = Image.open(file).convert("RGB")
    numpy_arr = np.array(pil_img)
    numpy_norm_arr = np.divide(numpy_arr, 255)
    return numpy_norm_arr


def reshape_image(image, shape, channels):
    reshape_size = shape.append(channels)
    return np.reshape(image, reshape_size)


# Accepts an array and number of categories, returns the one hot of the array
def get_one_hot(array, categories):
    array = np.array(array, dtype=np.int32)
    b = np.zeros((array.size, categories))
    b[np.arange(array.size), array] = 1
    return b


def get_disposable_dataset(dataset):
    return DisposableGenderDataset(dataset)


class GenderDataset:
    # Class Stuff
    def __init__(self):
        self.image_data = None
        self.gender_data = None
        self.size = 0

    def __iter__(self):
        return [self.image_data, self.gender_data]

    def __getitem__(self, key):
        return [self.image_data, self.gender_data][key]

    # Functions
    def get_first_image_pixel_count(self):
        print(np.shape(self.image_data))
        return np.shape(self.image_data)[0]

    def load_from_images(self, directory, shape, channels):
        images = []
        genders = []
        batch_size = 0

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for file in filenames:
                filepath = path.join(dirpath, file)
                image, gender = get_image_and_label_from_image_and_reshape(filepath, shape, channels)
                image = image.flatten()
                gender = gender.flatten()
                images.append(image)
                genders.append(gender)
                batch_size += 1

        self.image_data = np.array(images)
        self.gender_data = np.array(genders)
        self.size = batch_size

    def load_from_file(self, file_path, mode=None, ):
        data = np.load(file_path, mode, False, False)
        self.size = data['size'][0]
        self.gender_data = data['gender']
        self.image_data = data['image']

        assert np.shape(self.gender_data)[0] == self.size
        assert np.shape(self.image_data)[0] == self.size

    def save_to_file(self, file_path):
        np.savez(file_path, size=self.size, image=self.image_data, gender=self.gender_data)


class DisposableGenderDataset(GenderDataset):
    def __init__(self, dataset=None):
        super(DisposableGenderDataset, self).__init__()
        self.batch_offset = 0

        self.repeat_shuffle = False
        self.repetitions = 0
        self.repeat_until = 1

        if dataset is not None:
            self.image_data = np.copy(dataset.image_data)
            self.gender_data = np.copy(dataset.gender_data)
            self.size = dataset.size

    def shuffle(self):
        p = np.random.permutation(len(self.image_data))
        self.image_data = np.array(self.image_data[p])
        self.gender_data = np.array(self.gender_data[p])
        return self

    def repeat(self, epochs=-1, shuffle_on_repeat=False):
        self.repeat_until = epochs;
        self.repeat_shuffle = shuffle_on_repeat
        self.repetitions = 0
        return self

    def get_next_batch(self, batch_size):
        # Truncate if the next batch wont fit
        if self.repetitions == self.repeat_until:
            raise IndexError()

        result = GenderDataset()

        img = []
        gen = []
        offset = self.batch_offset
        for count in range(batch_size):
            if self.repetitions == self.repeat_until:
                batch_size = count

            # Add to batch
            index = count + offset
            img.append(self.image_data[index % self.size])
            gen.append(self.gender_data[index % self.size])

            # Increment Offset
            self.batch_offset += 1
            if self.batch_offset >= self.size:
                self.batch_offset -= self.size
                self.repetitions += 1
                if self.repeat_shuffle:
                    self.shuffle()

        result.image_data = np.array(img)
        result.gender_data = np.array(gen)
        result.size = batch_size

        return result
