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

import numpy as np
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

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), True, False)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False, False)
    
    def add_noise(self, img):
        noise_factor = 0.1
        noise = np.random.randn(*img.shape)
        img_noisy = img + noise_factor * noise
        return np.clip(img_noisy, 0.0, 255.0)

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """


        img = tf.keras.applications.vgg16.preprocess_input(img)
        img = self.add_noise(img)
        
        return img

    def get_data(self, path, shuffle, augment):
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
                preprocessing_function=self.preprocess_fn
                , 
                brightness_range=[0.8, 1.2],
                )

        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 224

        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='binary',
            batch_size=hp.batch_size,
            shuffle=shuffle
        )
        return data_gen