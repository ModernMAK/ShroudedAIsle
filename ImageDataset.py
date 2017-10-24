import numpy as np
import os
from os import path
from PIL import Image


# Returns a tuple of an image array and a one hot gender array
def get_image_and_label_from_file(file_path):
    # Get the gender from file path
    gender = get_gender_from_file_path(file_path)
    # Get image from file
    image = get_image_from_file(file_path)
    # Make one hot
    gender_one_hot = get_one_hot(gender, 2)
    return image, gender_one_hot


# Returns a tuple of an image array and a one hot gender array
def get_image_and_label_from_file_and_reshape(file_path, shape, channels):
    # Gets the image and gender (one hot)
    image, gender = get_image_and_label_from_file(file_path)
    # Reshapes the image
    image = reshape_image(image, shape, channels)
    return image, gender


def get_gender_from_file_path(file_path):
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


def get_image_from_file_and_reshape(file, shape, channels):
    image = get_image_from_file(file)
    image = reshape_image(image, shape, channels)
    return image


def reshape_image(image, shape, channels):
    reshape_size = shape.append(channels)
    return np.reshape(image, reshape_size)


# Accepts an array and number of categories, returns the one hot of the array
def get_one_hot(array, categories):
    array = np.array(array, dtype=np.int32)
    b = np.zeros((array.size, categories))
    b[np.arange(array.size), array] = 1
    return b


class ImageDataSet:
    # Class Stuff
    def __init__(self):
        self.image_data = None
        self.label_data = None
        self.size = 0

    def __iter__(self):
        return [self.image_data, self.label_data]

    def __getitem__(self, key):
        return [self.image_data, self.label_data][key]

    # Functions
    def get_first_image_pixel_count(self):
        print(np.shape(self.image_data))
        return np.shape(self.image_data)[0]

    def load_from_images(self, directory, shape, channels):
        images = []
        labels = []
        batch_size = 0

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for file in filenames:
                filepath = path.join(dirpath, file)
                image = get_image_from_file_and_reshape(filepath, shape, channels)
                label = filepath
                image = image.flatten()
                images.append(image)
                labels.append(label)
                batch_size += 1

        self.image_data = np.array(images)
        self.label_data = np.array(labels)
        self.size = batch_size

    def load_from_file(self, file_path, mode=None, ):
        data = np.load(file_path, mode, False, False)
        self.size = data['size'][0]
        self.label_data = data['gender']
        self.image_data = data['image']

        assert np.shape(self.label_data)[0] == self.size
        assert np.shape(self.image_data)[0] == self.size

    def save_to_file(self, file_path):
        np.savez(file_path, size=self.size, image=self.image_data, gender=self.label_data)


class ImageGenderDataSet(ImageDataSet):
    # Class Stuff
    def __init__(self):
        super().__init__()

    def load_from_images(self, directory, shape, channels):
        images = []
        genders = []
        batch_size = 0

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for file in filenames:
                filepath = path.join(dirpath, file)
                image, gender = get_image_and_label_from_file_and_reshape(filepath, shape, channels)
                image = image.flatten()
                gender = gender.flatten()
                images.append(image)
                genders.append(gender)
                batch_size += 1

        self.image_data = np.array(images)
        self.label_data = np.array(genders)
        self.size = batch_size


class DisposableImageDataSet(ImageDataSet):
    def __init__(self, data_set=None):
        super(DisposableImageDataSet, self).__init__()
        self.batch_offset = 0

        self.repeat_shuffle = False
        self.repetitions = 0
        self.repeat_until = 1

        if data_set is not None:
            self.image_data = np.copy(data_set.image_data)
            self.label_data = np.copy(data_set.label_data)
            self.size = data_set.size

    # resets batch offset and repititions
    def reset(self):
        self.repetitions = 0
        self.batch_offset = 0

    def shuffle(self):
        p = np.random.permutation(len(self.image_data))
        self.image_data = np.array(self.image_data[p])
        self.label_data = np.array(self.label_data[p])
        return self

    def repeat(self, epochs=-1, shuffle_on_repeat=False):
        self.repeat_until = epochs;
        self.repeat_shuffle = shuffle_on_repeat
        self.repetitions = 0
        return self

    def get_iterations_given_batch_size(self, batch_size):
        from math import ceil
        return ceil(self.size / batch_size)

    def get_next_batch(self, batch_size):
        # Truncate if the next batch wont fit
        if self.repetitions == self.repeat_until:
            raise IndexError()

        result = ImageGenderDataSet()

        img = []
        gen = []
        offset = self.batch_offset
        for count in range(batch_size):
            if self.repetitions == self.repeat_until:
                batch_size = count

            # Add to batch
            index = count + offset
            img.append(self.image_data[index % self.size])
            gen.append(self.label_data[index % self.size])

            # Increment Offset
            self.batch_offset += 1
            if self.batch_offset >= self.size:
                self.batch_offset -= self.size
                self.repetitions += 1
                if self.repeat_shuffle:
                    self.shuffle()

        result.image_data = np.array(img)
        result.label_data = np.array(gen)
        result.size = batch_size

        return result
