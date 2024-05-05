import os
import shutil
###########################################################################################################################
###################         THIS IS FOR MAKING COPIES OF DATA THAT ONLY HAVE 'STILL' IN THE FILE NAME #####################
###################         HOWEVER, MAYBE THE FACEFORENSICS PREPROCESS.PY FILE HAS ALL THE           #####################
###################         PREPROCESSING WE NEED SINCE IS DETECTS FACES AND THEN CREATES BOUNDING    #####################
###################         BOXES AROUND THE FACES AND CROPS IT. IDK WE'LL HAVE TO SEE WHAT WORKS     #####################

def ensure_dir(directory):
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def filter_and_copy_files(source_dir, target_dir, keyword):
    """Filter and copy files containing the keyword in their names from source to target directory."""
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if keyword in file:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(target_dir, file)
                shutil.copy2(src_file_path, dst_file_path)

def main():
    # Define directories
    base_data_dir = "data"
    filtered_data_dir = "filtered_data"

    original_source_dir = os.path.join(base_data_dir, "original_sequences/actors/c23/videos")
    manipulated_source_dir = os.path.join(base_data_dir, "manipulated_sequences/DeepFakeDetection/c23/videos")
    
    original_target_dir = os.path.join(filtered_data_dir, "originals")
    manipulated_target_dir = os.path.join(filtered_data_dir, "DeepFakes")

    # Ensure target directories exist
    ensure_dir(original_target_dir)
    ensure_dir(manipulated_target_dir)

    # Filter and copy original videos
    filter_and_copy_files(original_source_dir, original_target_dir, "still")

    # Filter and copy manipulated videos
    filter_and_copy_files(manipulated_source_dir, manipulated_target_dir, "still")

if __name__ == "__main__":
    main()

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):

        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        #self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), is_vgg=True, shuffle=True, augment=True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), is_vgg=True, shuffle=False, augment=True)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)

        # ==========================================================

        print("Dataset mean shape: [{0}, {1}, {2}]".format(
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[2]))

        print("Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0,0,0], self.mean[0,0,1], self.mean[0,0,2]))

        print("Dataset std shape: [{0}, {1}, {2}]".format(
            self.std.shape[0], self.std.shape[1], self.std.shape[2]))

        print("Dataset std top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0,0,0], self.std[0,0,1], self.std[0,0,2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
        img = (img - self.mean) / self.std    

        return img
    
    def add_noise(self, img):
        """ Function for adding noise to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)
        
        Returns:
            img_noisy - numpy array of shape (image size, image size, 3)
        """

        noise_factor = 0.1
        noise = np.random.randn(*img.shape)
        img_noisy = img + noise_factor * noise
        return np.clip(img_noisy, 0.0, 255.0)

    def add_black_squares_randomly(self, img):
        """ Function for adding black squares to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)
        
        Returns:
            img_black - numpy array of shape (image size, image size, 3)
        """

        min_square_size: int = 5
        max_square_size: int = 15
        img_with_squares = img.copy()
        for _ in range(np.random.randint(0, 5)): # Add 0-4 squares of size square_size between min_square_size and max_square_size
            square_size = np.random.randint(min_square_size, max_square_size)
            x = np.random.randint(0, hp.img_size - square_size)
            y = np.random.randint(0, hp.img_size - square_size)
            img_with_squares[x:x+square_size, y:y+square_size, :] = 0
        return img_with_squares

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = self.add_noise(img)
        img = self.add_black_squares_randomly(img)
        img = img / 255.
        img = self.standardize(img)
        # img = tf.keras.applications.vgg16.preprocess_input(img)
        return img

    def get_data(self, path, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                brightness_range=[0.9, 1.1],
                # rotation_range=3, 
                # width_shift_range=0.15, 
                # height_shift_range=0.15,
                # zoom_range=0.10,
                # horizontal_flip=True
                )

        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 224 if is_vgg else hp.img_size

        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='binary',
            batch_size=hp.batch_size,
            shuffle=shuffle
        )
        return data_gen