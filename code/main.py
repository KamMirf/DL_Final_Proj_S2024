import os
import sys
import argparse
from datetime import datetime
import requests
import tensorflow as tf
import re
import numpy as np


from preprocess import extract_single_face
import hyperparameters as hp
from model import VGGModel
from data_augment import Datasets
from tensorboard_utils import \
        CustomModelSaver
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
Note: When loading a checkpoint in, it only loads the weights for that process. Once it finishes, there will be NO weights loaded in
the head of the model. The VGG weights always get loaded in before.

New flag: --resume-training
    - when loading a checkpoint, set this flag to let the model continue training from the epoch specified in the weight filename
    
For comparing the model weights Jason: (Loads in baseline weights)
python3 main.py --load-checkpoint ../weights/baseline/vgg.e017-acc0.7418.weights.h5 --predict {image path or jpg link}
python3 main.py --load-checkpoint ../weights/baseline/vgg.e017-acc0.7418.weights.h5 --predict ../cropped_data/test/deepfake/01_02__meeting_serious__YVGY8LOK_frame_200_face_0.jpg
"""

""" 
Evaluate test data from a save:
Ex. python3 main.py --load-checkpoint checkpoints/vgg_model/043024-145118/vgg.weights.e000-acc0.5924.h5 --evaluate


Predict on a single image:
python3 main.py --predict {path to image}
Ex. python3 main.py --predict ../data/real_and_fake_face/split_data/test/real/real_00001.jpg
Also works with URLs:
Ex. python3 main.py --predict https://www.example.com/image.jpg
Can be used with --load-checkpoint to predict on a model from a save

LIME:
python main.py --evaluate --lime-image ../cropped_data/test/deepfake/01_02__outside_talking_still_laughing__YVGY8LOK_frame_300_face_0.jpg


python3 main.py --load-checkpoint {path} --lime-image {path}
"""

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        # default='..'+os.sep+'combined_data'+os.sep,
        default='..'+os.sep+'cropped_data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Indicates whether to resume training from the checkpoint epoch.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--predict',
        default=None,
        help='''Path to an image file to predict on. If this is
        provided, the model will load the checkpoint and make a
        prediction on the image.'''
    )
    parser.add_argument(
        '--lime-image',
        default=None,
        help='''Name of an image in the dataset to use for LIME evaluation.''')
    
    return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn, timestamp):

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """
    
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='epoch',
            profile_batch=0),
        CustomModelSaver(checkpoint_path, 3, hp.max_num_weights)
    ]

    if ARGS.resume_training:
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see data_augment.py get_data()
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )
    else:
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see data_augment.py get_data()
            callbacks=callback_list
        )


def test(model, test_data):
    """ Testing routine. """

    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0
    
    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of main.py
    os.chdir(sys.path[0])
    print("ARGS data: " + ARGS.data) 
    datasets = Datasets(ARGS.data)

    
    model = VGGModel()
    checkpoint_path = "checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    
    logs_path = "logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    if not ARGS.predict: # Don't print summaries if just predicting on an image
        model.vgg16.summary()
        model.head.summary()

    # Load base of VGG model
    model.vgg16.load_weights(ARGS.load_vgg)

    # Load checkpoints
    if ARGS.load_checkpoint is not None: 
        model.head.load_weights(ARGS.load_checkpoint) #new tf version doesn't use by_name=False

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.compile(
        optimizer=model.optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    if ARGS.predict:
        # Load image and make prediction
        if ARGS.predict.startswith("http"):
            img = requests.get(ARGS.predict, headers={'User-Agent': 'Mozilla/5.0'}).content
            img_path = "temp.jpg"
            with open(img_path, 'wb') as f:
                f.write(img)
        else:
            img_path = ARGS.predict
        new_path = extract_single_face(img_path)
        img = tf.io.read_file(new_path)
        if ARGS.predict.startswith("http"):
            os.remove(img_path) # Remove original temp file for URL
        os.remove(new_path) # Remove temp file
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_label = 'Real' if prediction > 0.5 else 'Fake'
        print(f"Model Prediction: {predicted_label}")
    elif ARGS.evaluate:
        test(model, datasets.test_data)
        path = ARGS.lime_image
        #LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    else:
        print("training")
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()