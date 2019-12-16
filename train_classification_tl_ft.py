import sys
import os.path
import argparse
import pickle

import numpy as np

import skimage
import skimage.io
import skimage.transform

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras import utils
from keras.models import Model, Sequential
import keras.models as models

from keras.layers import Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

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

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class CSVMetrics(CSVLogger):
    def set_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if epoch % args.epochs_ckpt == 0:
            prefixes = ["train", "val"]
            data_x = [self.x_train, self.validation_data[0]]
            data_y = [self.y_train, self.validation_data[1]]

            for prefix, x, y in zip(prefixes, data_x, data_y):
                metrics = calc_metrics(self.model, x, y, prefix=prefix)
                logs.update(metrics)

            if args.verbose:
                print(logs)

            super().on_epoch_end(epoch, logs)

def calc_metrics(model, x, y, prefix=""):
    results = {}

    y_pred = np.argmax(np.asarray(model.predict(x)), axis=-1)
    y_true = np.argmax(y, axis=-1)

    results[prefix + "_precision"] = precision_score(y_true, y_pred)
    results[prefix + "_recall"] = recall_score(y_true, y_pred)
    results[prefix + "_f1"] = f1_score(y_true, y_pred)

    results[prefix + "_tp"] = np.sum(y_true * y_pred)
    results[prefix + "_fp"] = np.sum((1 - y_true) * y_pred)
    results[prefix + "_tn"] = np.sum((1 - y_true) * (1 - y_pred))
    results[prefix + "_fn"] = np.sum(y_true * (1 - y_pred))

    return results


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
    parser.add_argument('--ckpt_load_weights', type=str, default=None)
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
    parser.add_argument('--fine_tuning', type=str2bool, default=False)
    parser.add_argument('--fine_tune_layers', type=int, default=4)

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
    image = skimage.img_as_float32(image)
    resized_image = skimage.transform.resize(image, (INPUT_WIDTH, INPUT_HEIGHT),
                                             anti_aliasing=True, mode='constant')
    # only keep 3 channels
    if resized_image.shape[2] != 3:
        resized_image = resized_image[:, :, 0:3]

    # extend data set by transforming data
    rotate_angles = [0, 90, 180, 270]
    # rotate_angles = [0, 180]
    images = [skimage.transform.rotate(resized_image, angle) for angle in rotate_angles]

    # normalize pictures?
    return images

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
        if not args.skip_train:
            with open(args.train_set, "rb") as f:
                train = pickle.load(f)
        else:
            train = {"0": [], "1": []}

        if not args.skip_test:
            with open(args.test_set, "rb") as f:
                test = pickle.load(f)
        else:
            test = {"0": [], "1": []}

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


def build_model(old_model):
    """
        Builds the new model by replacing the last layer of old_model with
         a new trainable binary layer (softmax activation)
    Args:
        old_model: old inception model

    Returns:
        New model with trainable last layer
    """

    # freeze old layers
    if args.fine_tuning:
        for layer in old_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False
    else:
        for layer in old_model.layers:
            layer.trainable = False

    # get relevant layers from old model
    inception_output = old_model.get_layer(index=-2).output

    if args.fine_tuning:
        inception_output = Dense(1024, activation='relu')(inception_output)
        inception_output = Dropout(0.5)(inception_output)

    # create new layer
    swisspv_prediction = Dense(NUM_CLASSES, activation='softmax')(inception_output)

    # build new model
    new_model = Model(inputs=old_model.input, outputs=swisspv_prediction)

    # save new model
    path = os.path.join(SAVE_DIR, f"keras_model_untrained_ft_{args.fine_tune_layers}.h5")
    new_model.save(path)

    return new_model


def run():
    # load model
    model = models.load_model(os.path.join(LOAD_DIR, args.ckpt_load), compile=False)

    # transform model if needed
    if args.from_scratch:
        model = build_model(model)

    # transform model to use multiple GPUs
    try:
        parallel_model = utils.multi_gpu_model(model)
        if args.verbose:
            print("Using multithreading")
    except Exception:
        parallel_model = model
        if args.verbose:
            print("Using multithreading failed")

        pass

    # show summary
    if args.verbose:
        print("Using model")
        parallel_model.summary()

    # load weights
    if args.ckpt_load_weights:
        print("Loading weights")
        parallel_model.load_weights(args.ckpt_load_weights, by_name=True)

    # compile model
    parallel_model.compile(optimizer=args.optimizer,
                           loss=args.loss,
                           metrics=['accuracy'])

    # load and split data
    x_train, y_train, x_test, y_test = load_data(shuffle=True)

    # fit model
    if not args.skip_train:
        # build label matrix
        y = utils.to_categorical(y_train, num_classes=NUM_CLASSES)

        # custom callback
        metrics = CSVMetrics(f"log_classification_ft_{args.fine_tune_layers}.csv")
        metrics.set_data(x_train, y)

        # fit model
        parallel_model.fit(x_train, y,
                           callbacks=[
                               metrics,
                               ModelCheckpoint(f"weights_classification_ft_{args.fine_tune_layers}.hdf5", monitor='val_loss',
                                               verbose=1, save_best_only=True, save_weights_only=True)
                           ],
                           epochs=args.epochs,
                           batch_size=args.batch_size,
                           validation_split=args.validation_split,
                           shuffle=True,
                           verbose=args.verbose)

        # build model name and save model
        model.save(os.path.join(SAVE_DIR, f"keras_model_trained_ft_{args.fine_tune_layers}.h5"))

    elif args.verbose:
        print("Skipping training")

    # eval model
    if not args.skip_test:
        print(f"Test on {len(y_test)} samples")

        # transform label list into label matrix
        y = utils.to_categorical(y_test, num_classes=NUM_CLASSES)

        # evaluate model
        score = parallel_model.evaluate(x_test, y,
                                        batch_size=args.batch_size,
                                        verbose=args.verbose)

        # compute metrics
        metrics = calc_metrics(parallel_model, x_test, y, prefix='test')
        for name, s in zip(parallel_model.metrics_names, score):
            metrics[name] = s

        print("\nResults")
        for metric in metrics:
            print(f"{metric}: {metrics[metric]}")

    elif args.verbose:
        print("Skipping testing")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # use some small values to test model
        sys.argv += [
            "--ckpt_load=keras_swisspv_untrained.h5",
            # "--ckpt_load_weights=weights_classification.hdf5",

            "--optimizer=rmsprop",
            "--loss=binary_crossentropy",

            "--from_scratch=True",
            "--skip_train=False",
            "--skip_test=False",

            "--epochs=1000",
            "--epochs_ckpt=5",
            "--batch_size=128",
            "--train_set=train.pickle",
            "--test_set=test.pickle",
            "--validation_split=0.25",

            "--fine_tuning=False",
            "--fine_tune_layers=4",

            "--verbose=1"
        ]

    args = parse_args()

    run()
