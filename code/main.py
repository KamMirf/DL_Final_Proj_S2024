import os
import sys
import argparse
from datetime import datetime
import requests
import tensorflow as tf
import re

from preprocess import extract_single_face
import hyperparameters as hp
from model import VGGModel
from data_augment import Datasets
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from matplotlib import pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" 
Continue training from a save: python3 main.py --load-checkpoint {path to h5 file in checkpoints/vgg_model/your model number}
Ex. python3 main.py --load-checkpoint checkpoints/vgg_model/043024-145118/vgg.weights.e000-acc0.5924.h5
Notes: 
- continues from epoch you left off on
- automatically saves checkpoint from epochs that produce a higher accuracy

Evaluate test data from a save:
Ex. python3 main.py --load-checkpoint checkpoints/vgg_model/043024-145118/vgg.weights.e000-acc0.5924.h5 --evaluate

Predict on a single image:
python3 main.py --predict {path to image}
Ex. python3 main.py --predict ../data/real_and_fake_face/split_data/test/real/real_00001.jpg
Also works with URLs:
Ex. python3 main.py --predict https://www.example.com/image.jpg
Can be used with --load-checkpoint to predict on a model from a save
"""

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '--task',
    #     required=True,
    #     choices=['1', '3'],
    #     help='''Which task of the assignment to run -
    #     training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
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
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
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

    return parser.parse_args()



def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """
    
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='epoch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, 3, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see data_augment.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
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


    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of main.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of main.py
    os.chdir(sys.path[0])
    print("ARGS data: " + ARGS.data) 
    datasets = Datasets(ARGS.data, 3)

    
    model = VGGModel()
    checkpoint_path = "checkpoints" + os.sep + \
        "vgg_model" + os.sep + timestamp + os.sep
    
    logs_path = "logs" + os.sep + "vgg_model" + \
        os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))

    # Print summaries for both parts of the model
    if not ARGS.predict: # Don't print summaries if just predicting on an image
        model.vgg16.summary()
        model.head.summary()

    # Load base of VGG model
    model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Load checkpoints
    if ARGS.load_checkpoint is not None: 
        model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    if ARGS.predict:
        # Load image and make prediction
        if ARGS.predict.startswith("http"):
            img = requests.get(ARGS.predict, headers={'User-Agent': 'Mozilla/5.0'}).content
            img_path = "temp.jpg"
            with open(img_path, 'wb') as f:
                f.write(img)
        else:
            img_path = ARGS.predict
        img_path = extract_single_face(img_path)
        img = tf.io.read_file(img_path)
        if ARGS.predict.startswith("http"):
            os.remove(img_path) # Remove temp file
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.expand_dims(img, axis=0)
        prediction = model.predict(img)
        if prediction[0][0] > prediction[0][1]:
            print("Model Prediction: Real")
        else:
            print("Model Prediction: Fake")
    elif ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        print("training")
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()