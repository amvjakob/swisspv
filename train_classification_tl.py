import sys
import os.path
import argparse
import pickle

import numpy as np

import skimage
import skimage.io
import skimage.transform

from keras import utils
from keras.models import Model
import keras.models as models

from keras.layers import Dense
from keras.callbacks import Callback

# log info
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# constants for files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = '/work/hyenergy/raw/SwissTopo/RGB_25cm/data_resized/crop_tool/classification'
PV_DIR = 'PV'
NO_PV_DIR = 'noPV'

LOAD_DIR = os.path.join(BASE_DIR, 'ckpt', 'inception_tl_load')
SAVE_DIR = os.path.join(BASE_DIR, 'ckpt', 'inception_tl_save')

# constants for images
INPUT_WIDTH = 299
INPUT_HEIGHT = 299

# constants for model
NUM_CLASSES = 2
TRAIN_TEST_SPLIT = 0.7

# used for standardization of pictures
INPUT_MEAN = 127.5
INPUT_STD = 127.5


# custom saver class for model
class SwissPVSaver(Callback):
    @staticmethod
    def build_model_path(epoch):
        model_name = f"keras_swisspv_" \
                     f"{epoch}_" \
                     f"optimizer={args.optimizer}_" \
                     f"loss={args.loss}_" \
                     f"batchsize={args.batch_size}" \
                     f".h5"

        return os.path.join(SAVE_DIR, model_name)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if epoch % args.epochs_ckpt == 0:  # save on each kth epoch
            self.model.save(SwissPVSaver.build_model_path(epoch))



def parse_args():
    """
        Parses the args given by the user
    Returns:
        The parsed args
    """

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # helper function to pass booleans
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_scratch', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--ckpt_load', type=str, default='keras_swisspv_untrained.h5')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epochs_ckpt', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--train_set', type=str, default='train.pickle')
    parser.add_argument('--test_set', type=str, default='test.pickle')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--skip_train', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--skip_test', type=str2bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()
    return args


def load_image(path):
    """
        Loads and transforms an image
    Args:
        path: path to the image

    Returns:
        a list of transforms of the given image
    """
    image = skimage.io.imread(path)
    resized_image = skimage.transform.resize(image, (INPUT_WIDTH, INPUT_HEIGHT),
                                             anti_aliasing=True, mode='constant')
    # only keep 3 channels
    if resized_image.shape[2] != 3:
        resized_image = resized_image[:, :, 0:3]

    # extend data set by transforming data
    # rotate_angles = [0, 90, 180, 270]
    rotate_angles = [0, 180]
    images = [skimage.transform.rotate(resized_image, angle) for angle in rotate_angles]

    # normalize pictures?
    return images


def build_model(old_model):
    """
        Builds the new model by replacing the last layer of old_model with
         a new trainable binary layer (softmax activation)
    Args:
        old_model: old inception model

    Returns:
        New model with trainable last layer
    """
    # decompose old model by removing last layer
    bottleneck_input = old_model.get_layer(index=0).input
    bottleneck_output = old_model.get_layer(index=-2).output
    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

    # freeze old layers
    for layer in bottleneck_model.layers:
        layer.trainable = False

    # build new model
    new_model = Sequential()
    new_model.add(bottleneck_model)
    if args.verbose:
        print('Model summary before final layer addition:', new_model.summary())

    # number of nodes in the second to last layer of the pre-trained model.
    input_dim = old_model.get_layer(index=-2).output.shape[1]
    if hasattr(input_dim, 'value'):
        input_dim = int(input_dim.value or 0)
    else:
        input_dim = int(input_dim)

    new_model.add(Dense(NUM_CLASSES, input_dim=input_dim,
                        activation='softmax'))  # convert outputs to probabilities

    if args.verbose:
        print('Model summary after final layer addition:', new_model.summary())

    # save new model
    path = os.path.join(SAVE_DIR, 'keras_model_untrained.h5')
    new_model.save(path)

    return new_model


def load_data(shuffle=True):
    """
        Load all images and labels

    Returns:
        x_train: list of images to be used for training
        y_train: list of training labels
        x_test: list of images to be used for testing
        y_train: list of testing labels
    """

    if os.path.exists(args.train_set) and os.path.exists(args.test_set):
        if args.verbose:
            print("Loading data from pickle files")

        # load data from pickle files
        with open(args.train_set, "rb") as f:
            train = pickle.load(f)

        with open(args.test_set, "rb") as f:
            test = pickle.load(f)

        return load_from_filenames(train, test, shuffle=shuffle)

    else:
        if args.verbose:
            print("Loading data from directory and generating pickle files")

        # load all filenames
        neg_dir = os.path.join(IMG_DIR, NO_PV_DIR)
        pos_dir = os.path.join(IMG_DIR, PV_DIR)

        # positive
        pos_imgs_names = []
        for filename in os.listdir(pos_dir):
            pos_imgs_names.append(filename)

        # negative
        neg_imgs_names = []
        for filename in os.listdir(neg_dir):
            neg_imgs_names.append(filename)

        # shuffle filenames
        np.random.shuffle(pos_imgs_names)
        np.random.shuffle(neg_imgs_names)

        # split according to ratio
        split_neg = int(TRAIN_TEST_SPLIT * len(neg_imgs_names))
        split_pos = int(TRAIN_TEST_SPLIT * len(pos_imgs_names))

        train = {
            "0": neg_imgs_names[:split_neg],
            "1": pos_imgs_names[:split_pos]
        }
        test = {
            "0": neg_imgs_names[split_neg:],
            "1": pos_imgs_names[split_pos:]
        }

        # save files as pickle
        with open("train.pickle", "wb") as f:
            pickle.dump(train, f)

        with open("test.pickle", "wb") as f:
            pickle.dump(test, f)

        return load_from_filenames(train, test, shuffle=shuffle)


def load_from_filenames(train, test, shuffle):
    """
    Loads examples and labels given their filenames

    Args:
        train: object containing filenames of training examples
        test: object containing filenames of testing examples
        shuffle: shuffles both datasets if True

    Returns:
        x_train: list of images to be used for training
        y_train: list of training labels
        x_test: list of images to be used for testing
        y_train: list of testing labels
    """
    classes = [0, 1]
    neg_dir = os.path.join(IMG_DIR, NO_PV_DIR)
    pos_dir = os.path.join(IMG_DIR, PV_DIR)
    dirs = [neg_dir, pos_dir]

    x_train, y_train = [], []
    x_test, y_test = [], []

    for class_, dir in zip(classes, dirs):
        for data, x, y in zip([train, test], [x_train, x_test], [y_train, y_test]):
            for name in data[str(class_)]:
                path = os.path.join(dir, name)
                images = load_image(path)
                x.extend(images)
                y.extend(np.repeat([class_], len(images)))

    # transform to numpy array
    x_train = np.asarray(x_train).reshape((len(x_train), INPUT_WIDTH, INPUT_HEIGHT, 3))
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test).reshape((len(x_test), INPUT_WIDTH, INPUT_HEIGHT, 3))
    y_test = np.asarray(y_test)

    # shuffle data
    if shuffle:
        p_train = np.random.permutation(len(y_train))
        p_test = np.random.permutation(len(y_test))
        return x_train[p_train], y_train[p_train], x_test[p_test], y_test[p_test]
    else:
        return x_train, y_train, x_test, y_test


def fit(model, imgs, labels):
    """
    Retrain the new model on the given data

    Args:
        new_model: Keras model to train
        imgs: training data
        labels: training labels

    Returns:
        the trained model

    """
    # binary classification problem
    # change params here according to deepsolar?

    try:
        model = utils.multi_gpu_model(model)
        print("Using multithreading")
    except Exception:
        print("Using multithreading failed")
        pass

    model.compile(optimizer=args.optimizer,
                  loss=args.loss,
                  metrics=['accuracy'])

    # transform label list into label matrix
    labels_as_matrix = utils.to_categorical(labels, num_classes=NUM_CLASSES)

    # build saver
    saver = SwissPVSaver()

    # fit model
    model.fit(imgs, labels_as_matrix,
              callbacks=[saver],
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=args.validation_split,
              shuffle=True,
              verbose=args.verbose)

    return model


def test(model, imgs, labels):
    """
    Test the model
    Args:
        model: model to evaluate
        imgs: test images
        labels: test labels

    Returns:

    """
    # transform label list into label matrix
    labels_as_matrix = utils.to_categorical(labels, num_classes=NUM_CLASSES)

    return model.evaluate(imgs, labels_as_matrix,
                          batch_size=args.batch_size,
                          verbose=args.verbose)


def run():
    # load model
    model = models.load_model(os.path.join(LOAD_DIR, args.ckpt_load))

    # transform model if needed
    if args.from_scratch:
        model = build_model(model)

    # load and split data
    x_train, y_train, x_test, y_test = load_data(shuffle=True)

    # fit model
    if not args.skip_train:
        model = fit(model, x_train, y_train)

        # build model name and save model
        model.save(SwissPVSaver.build_model_path(-1))
    elif args.verbose:
        print("Skipping training")

    # eval model
    if not args.skip_test:
        score = test(model, x_test, y_test)

        print("\nResults")
        for name, s in zip(model.metrics_names, score):
            print(f"{name}: {s}")

        # compute stats
        stats = [0, 0, 0, 0]  # TP, TN, FP, FN
        for x, y in zip(x_test, y_test):
            # could be that the predictions here are the inverse of the true label
            prediction = np.argmax(model.predict(x.reshape(1, INPUT_WIDTH, INPUT_HEIGHT, 3))[0])
            if prediction == 1 and y == 1:
                stats[0] += 1
            elif prediction == 0 and y == 0:
                stats[1] += 1
            elif prediction == 1 and y == 0:
                stats[2] += 1
            elif prediction == 0 and y == 1:
                stats[3] += 1

        print("TP: {}, TN: {}, FP: {}, FN: {}".format(*stats))

    elif args.verbose:
        print("Skipping testing")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # use some small values to test model
        sys.argv += [
            "--ckpt_load=keras_model_untrained.h5",

            "--optimizer=rmsprop",
            "--loss=binary_crossentropy",

            "--from_scratch=False",
            "--skip_train=False",
            "--skip_test=False",

            "--epochs=1000",
            "--epochs_ckpt=10",
            "--batch_size=100",
            "--train_set=train_0_7.pickle",
            "--test_set=test_0_7.pickle",
            "--validation_split=0.25",

            "--verbose=1"
        ]

    args = parse_args()

    run()
