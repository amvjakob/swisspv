import sys
import os.path
import argparse
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import skimage
import skimage.io
import skimage.transform

from sklearn.metrics import f1_score, precision_score, recall_score

from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from keras.utils import to_categorical, multi_gpu_model
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

# log info
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# constants for files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PV_DIR = 'PV'
NO_PV_DIR = 'noPV'

LOAD_DIR = os.path.join(BASE_DIR, 'ckpt', 'inception_tl_load')
SAVE_DIR = os.path.join(BASE_DIR, 'ckpt', 'inception_tl_save')

# constants for images
INPUT_WIDTH = 299
INPUT_HEIGHT = 299

# constants for model
NUM_CLASSES = 2
TRAIN_TEST_SPLIT = 0.9

# Constants dictating the learning rate schedule.
LR_INITIAL = 0.001
EPOCHS_PER_DECAY = 5
LR_DECAY = 0.5
RMSPROP_DECAY = 0.8  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 0.1  # Epsilon term for RMSProp.

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9998
BATCHNORM_EPSILON = 0.001
LAYER_REG = 0.00004

# imbalanced rate = alpha + 1 (loss penalty on minority class)
ALPHA = 3

# label smoothing
LABEL_SMOOTHING = 0.1


# Helper class for logging of custom metrics
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

    pred = np.asarray(model.predict(x))
    y_pred = np.argmax(pred, axis=-1)
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
    parser.add_argument('--ckpt_load', type=str, default='keras_swisspv_untrained.h5')
    parser.add_argument('--ckpt_load_weights', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=1)
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
    parser.add_argument('--fine_tune_layers', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/work/hyenergy/raw/SwissTopo/RGB_25cm/data_resized/crop_tool/classification')

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
        neg_dir = os.path.join(args.data_dir, NO_PV_DIR)
        pos_dir = os.path.join(args.data_dir, PV_DIR)

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
    neg_dir = os.path.join(args.data_dir, NO_PV_DIR)
    pos_dir = os.path.join(args.data_dir, PV_DIR)
    dirs = [neg_dir, pos_dir]

    x_train, y_train = [], []
    x_test, y_test = [], []

    DEBUG = False
    if DEBUG:
        classes = reversed(classes)
        dirs = reversed(dirs)

    for class_, dir in zip(classes, dirs):
        for data, x, y in zip([train, test], [x_train, x_test], [y_train, y_test]):
            for name in data[str(class_)]:
                path = os.path.join(dir, name)
                images = load_image(path)

                x.extend(images)
                y.extend(np.repeat([class_], len(images)))

                if DEBUG: break
            # if DEBUG: break
        if DEBUG: break

    # transform to numpy array
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # normalize images
    np.subtract(x_train, 0.5)
    np.multiply(x_train, 2)
    np.subtract(x_test, 0.5)
    np.multiply(x_test, 2)

    # reshape
    x_train = x_train.reshape((len(x_train), INPUT_WIDTH, INPUT_HEIGHT, 3))
    x_test = x_test.reshape((len(x_test), INPUT_WIDTH, INPUT_HEIGHT, 3))



    # shuffle data
    if shuffle:
        p_train = np.random.permutation(len(y_train))
        p_test = np.random.permutation(len(y_test))
        return x_train[p_train], y_train[p_train], x_test[p_test], y_test[p_test]
    else:
        return x_train, y_train, x_test, y_test


def run():
    # load model
    model = load_model(os.path.join(LOAD_DIR, args.ckpt_load), compile=False)

    # modify model
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.kernel_regularizer = l2(LAYER_REG)

        if isinstance(layer, BatchNormalization):
            layer.momentum = BATCHNORM_MOVING_AVERAGE_DECAY
            layer.epsilon = BATCHNORM_EPSILON

    # define exponential decay for learning rate
    def lr_decay_callback(decay_rate, decay_steps, use_staircase=True):
        decay_rate_smooth = np.power(decay_rate, 1.0/decay_steps)
        def step_decay(epoch, lr):
            if use_staircase:
                if epoch % decay_steps == 0:
                    return lr * decay_rate
                return lr
            else:
                return lr * decay_rate_smooth

        return LearningRateScheduler(step_decay)

    lr_decay = lr_decay_callback(LR_DECAY, EPOCHS_PER_DECAY)
    # find a way to use momentum param?
    optimizer = Adam(lr=LR_INITIAL,
                                decay=RMSPROP_DECAY,
                                epsilon=RMSPROP_EPSILON)


    # categorical crossentropy as loss function
    def build_loss(label_smoothing=0.0):

        def loss_fn(y_true, y_pred):

            labels = K.cast(np.argmax(y_true, axis=-1), dtype='int64')
            penalty_vector = np.add(np.multiply(K.cast(ALPHA, dtype='int64'), labels), 1)
            penalty_vector = K.cast(penalty_vector, dtype='float32')

            # label smoothing
            if label_smoothing > 0.0:
                smooth_positives = 1.0 - label_smoothing
                smooth_negatives = label_smoothing / NUM_CLASSES
                smooth_labels = y_true * smooth_positives + smooth_negatives
            else:
                smooth_labels = y_true

            loss = categorical_crossentropy(smooth_labels, y_pred)

            cost_sensitive_cross_entropy = np.multiply(penalty_vector, loss)
            return K.mean(cost_sensitive_cross_entropy)

        return loss_fn

    """
    # freeze layers
    if not args.skip_train:
        if args.fine_tune_layers:
            for layer in model.layers[:-args.fine_tune_layers]:
                layer.trainable = False
        else:
            for layer in model.layers[:-2]:
                # last two layers are Dense and Softmax
                layer.trainable = False
    """

    # transform model to use multiple GPUs
    try:
        parallel_model = multi_gpu_model(model)
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
    parallel_model.compile(optimizer=optimizer,
                           loss=build_loss(label_smoothing=LABEL_SMOOTHING),
                           metrics=['accuracy'])

    # load and split data
    x_train, y_train, x_test, y_test = load_data(shuffle=True)

    # fit model
    if not args.skip_train:
        # build label matrix
        y = to_categorical(y_train, num_classes=NUM_CLASSES)

        # custom callback
        metrics = CSVMetrics(f"log_classification_{args.fine_tune_layers}.csv")
        metrics.set_data(x_train, y)

        # fit model
        parallel_model.fit(x_train, y,
                           callbacks=[
                               metrics,
                               lr_decay,
                               ModelCheckpoint(f"weights_classification_{args.fine_tune_layers}.hdf5",
                                               monitor='val_loss', verbose=1, save_best_only=True,
                                               save_weights_only=True)
                           ],
                           epochs=args.epochs,
                           batch_size=args.batch_size,
                           validation_split=args.validation_split,
                           shuffle=True,
                           verbose=args.verbose)

        # build model name and save model
        model.save(os.path.join(SAVE_DIR, f"keras_model_trained_{args.fine_tune_layers}.h5"))

    elif args.verbose:
        print("Skipping training")

    # eval model
    if not args.skip_test:
        print(f"Test on {len(y_test)} samples")

        # transform label list into label matrix
        y = to_categorical(y_test, num_classes=NUM_CLASSES)

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

            "--skip_train=False",
            "--skip_test=False",

            "--fine_tune_layers=2",

            "--epochs=10",
            "--epochs_ckpt=5",
            "--batch_size=100",
            "--train_set=train.pickle",
            "--test_set=test.pickle",
            "--validation_split=0.1",

            "--verbose=1"
        ]

    args = parse_args()

    run()
