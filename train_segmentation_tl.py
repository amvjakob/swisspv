import sys
import os.path
import argparse
import pickle

import numpy as np

import skimage
import skimage.io
import skimage.transform

from tensorflow import reduce_mean as tf_reduce_mean
from tensorflow import matmul, image, gather, transpose, reshape

from keras import utils, initializers, optimizers, callbacks, layers, losses, metrics
from keras.models import Model, load_model

from keras import backend as K

# constants for files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PV_DIR = 'PV'
NO_PV_DIR = 'noPV'

RESULT_DIR = 'segmentation_results'

#LOAD_DIR = os.path.join('/scratch/dhaene', 'ckpt', 'inception_tl_load')
#SAVE_DIR = os.path.join('/scratch/dhaene', 'ckpt', 'inception_tl_save')
LOAD_DIR = os.path.join('ckpt', 'seg_load')
SAVE_DIR = os.path.join('ckpt', 'seg_save')

# constants for images
INPUT_WIDTH = 299
INPUT_HEIGHT = 299

# constants for model
NUM_CLASSES = 2
TRAIN_TEST_SPLIT = 0.9

SEGMENTATION_THRES = 0.37 # threshold for segmenting solar panel

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
    parser.add_argument('--build_model', type=str2bool, nargs='?',
                        const=True, default=False)
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
    parser.add_argument('--second_layer_from_ckpt', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--two_layers', type=str2bool, nargs='?',
                        const=True, default=False)

    parser.add_argument('--seg_1_weights', type=str, default='seg_1_weights.hdf5')
    parser.add_argument('--seg_2_weights', type=str, default='seg_2_weights.hdf5')
    parser.add_argument('--data_dir', type=str,
                        default='/work/hyenergy/raw/SwissTopo/RGB_25cm/data_resized/crop_tool/classification')

    args = parser.parse_args()
    return args

def load_image(path):
    """
        Loads and transforms an image as float32.
        Resize image to dimensions INPUT_WIDTH x INPUT_HEIGHT.
        Rotate image by 0, 90, 180 and 270 degrees.

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
    images = [skimage.transform.rotate(resized_image, angle) for angle in rotate_angles]

    return images

def rescale_CAM(classmap_val):
    # rescale class activation map to [0, 1].
    CAM_rescale = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_val))
    CAM_rescale = CAM_rescale[0]
    return CAM_rescale

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

    for class_, dir in zip(classes, dirs):
        for data, x, y in zip([train, test], [x_train, x_test], [y_train, y_test]):
            for name in data[str(class_)]:
                path = os.path.join(dir, name)
                images = load_image(path)
                x.extend(images)
                y.extend(np.repeat([class_], len(images)))

    # transform to numpy
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # normalize images
    np.subtract(x_train, 0.5)
    np.multiply(x_train, 2)
    np.subtract(x_test, 0.5)
    np.multiply(x_test, 2)

    # shuffle data
    if shuffle:
        p_train = np.random.permutation(len(y_train))
        p_test = np.random.permutation(len(y_test))
        x_train, y_train, x_test, y_test = x_train[p_train], y_train[p_train], x_test[p_test], y_test[p_test]

    # transform to numpy array
    x_train = np.reshape(x_train, (len(x_train), INPUT_WIDTH, INPUT_HEIGHT, 3))
    x_test = np.reshape(x_test, (len(x_test), INPUT_WIDTH, INPUT_HEIGHT, 3))

    return x_train, y_train, x_test, y_test

def build_model():
    # load old model
    old_model = load_model(os.path.join(LOAD_DIR, args.ckpt_load), compile=False)

    model_input = old_model.get_layer(index=0).input
    feature_map = old_model.get_layer(name='mixed2').output

    # freeze old layers
    for layer in old_model.layers:
        layer.trainable = False

    # create first layer to add
    conv = layers.Conv2D(512, (3, 3),
                         strides=(1, 1),
                         name='conv_aux_1',
                         padding='same',
                         use_bias=True,
                         trainable=not args.two_layers,
                         bias_initializer=initializers.Constant(value=0.1),
                         activation='relu',
                         kernel_initializer=initializers.TruncatedNormal(stddev=1e-4))(feature_map)

    if args.two_layers:
        # create second layer to add
        conv = layers.Conv2D(512, (3, 3),
                             strides=(1, 1),
                             name='conv_aux_2',
                             padding='same',
                             use_bias=True,
                             bias_initializer=initializers.Constant(value=0.1),
                             activation='relu',
                             kernel_initializer=initializers.TruncatedNormal(stddev=1e-4))(conv)

    # global average pool
    GAP = layers.Lambda(lambda x: tf_reduce_mean(x, axis=[1,2], keepdims=False), name="GAP")(conv)

    # linear model
    logits = layers.Dense(NUM_CLASSES,
                          name='W',
                          use_bias=False,
                          activation='softmax',
                          kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.01))(GAP)

    # create new model
    new_model = Model(inputs=model_input, outputs=logits)
    if args.verbose > 1:
        new_model.summary()

    try:
        parallel_model = utils.multi_gpu_model(new_model)
        print("Using multithreading")
    except Exception:
        parallel_model = new_model
        print("Using multithreading failed")
        pass

    if args.two_layers:
        if args.second_layer_from_ckpt:
            new_model.load_weights("seg_2_weights.hdf5", by_name=True)
        else:
            new_model.load_weights("seg_1_weights.hdf5", by_name=True)

    parallel_model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])

    # save new model
    filename = f'keras_seg_base_{"1" if not args.two_layers else "2"}.h5'
    path = os.path.join(LOAD_DIR, filename) # save in load dir
    new_model.save(path)

    return new_model, parallel_model


def fit(model, imgs, labels):

    # transform label list into label matrix
    x = np.reshape(imgs, [len(imgs), INPUT_WIDTH, INPUT_HEIGHT, 3])

    weights_name = f'seg_{"1" if not args.two_layers else "2"}_weights.hdf5'

    # fit model
    model.fit(x, labels,
              callbacks=[callbacks.ModelCheckpoint(weights_name,
                                                   verbose=1,
                                                   save_weights_only=True,
                                                   period=args.epochs_ckpt)],
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=args.validation_split,
              shuffle=True,
              verbose=args.verbose)



    return model

def test(model, x_test, y_test):
    input = model.get_layer(index=0).input

    layer_name = f'conv_aux_{"1" if not args.two_layers else "2"}'
    conv_map = model.get_layer(name=layer_name)
    conv_reshape = layers.Lambda(lambda x: image.resize_bilinear(x, [100, 100]), name="conv_reshape")(conv_map.output)
    conv_reshape_2 = layers.Lambda(lambda x: reshape(x, [-1, 100 * 100, 512]), name="conv_reshape_2")(conv_reshape)

    w_layer = model.get_layer(name="W")
    w_layer_weights = w_layer.get_weights()
    w_layer_weights = w_layer_weights[0]
    w_c = transpose(w_layer_weights)
    w_c = reshape(w_c, [-1, 512, 1])

    # we have two channels in the CAM
    # by taking the channel 1, we see the activation map of 'PV'
    # by taking the channel 0, we see the activation map of 'noPV'
    cam = layers.Lambda(lambda x: reshape(matmul(x, w_c), [-1, 100, 100])[1:2], name="cam")(conv_reshape_2)

    test_model = Model(inputs=input, outputs=cam)
    test_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    stats = np.array([0, 0, 0])
    area_error = []
    estimate_total_area = 0
    true_total_area = 0

    # check that folders exist
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
        os.mkdir(os.path.join(RESULT_DIR, 'TP'))
        os.mkdir(os.path.join(RESULT_DIR, 'FP'))
    elif not os.path.exists(os.path.join(RESULT_DIR, 'TP')):
        os.mkdir(os.path.join(RESULT_DIR, 'TP'))
    elif not os.path.exists(os.path.join(RESULT_DIR, 'FP')):
        os.mkdir(os.path.join(RESULT_DIR, 'FP'))

    yhat = model.predict(x_test, batch_size=args.batch_size, verbose=args.verbose)
    predictions = np.argmax(yhat, axis=1)
    for i, prediction in enumerate(predictions):

        if prediction == 1 and yhat[prediction] > 0.5:
            x = np.reshape(x_test[i], [1, INPUT_WIDTH, INPUT_HEIGHT, 3])
            CAM = test_model.predict(x, batch_size=1)
            CAM = rescale_CAM(CAM)

            pred_pixel_area = np.sum(CAM > SEGMENTATION_THRES)  # predicted/estimated pixel area
            estimate_total_area += pred_pixel_area

            CAM_img = np.array(CAM) # np.asarray([[int(255 * val) for val in row] for row in CAM])

            if y_test[i] == 0:  # FP
                stats[1] += 1
                # save original image and CAM.
                skimage.io.imsave(
                    os.path.join(RESULT_DIR, 'FP', str(i) + '_original.png'),
                    x_test[i])
                skimage.io.imsave(
                    os.path.join(RESULT_DIR, 'FP', str(i) + '_CAM.png'),
                    CAM_img)

            else:  # TP
                stats[0] += 1

                # save original image and CAM.
                skimage.io.imsave(
                    os.path.join(RESULT_DIR, 'TP', str(i) + '_original.png'),
                    x_test[i])
                skimage.io.imsave(
                    os.path.join(RESULT_DIR, 'TP', str(i) + '_CAM.png'),
                    CAM_img)

                # compare with ground truth segmentation.
                """
                true_seg_img = skimage.io.imread(
                    os.path.join(FLAGS.eval_set_dir, os.path.splitext(os.path.basename(img_path))[0] + '_true_seg.png'))
                true_seg_img = true_seg_img.astype(np.float64)
                np.divide(true_seg_img, 255.0, out=true_seg_img)
                true_pixel_area = np.sum(true_seg_img)
                true_pixel_area = true_pixel_area * (RESOLUTION * RESOLUTION) / (
                            ORIGINAL_IMAGE_SIZE * ORIGINAL_IMAGE_SIZE)
                true_total_area += true_pixel_area
                area_error.append(true_pixel_area - pred_pixel_area)
                """
                '''
                pred =
                result = dice_coef(true_seg_img, pred)
                result_value = result.eval()
                IoU.append(result_value)
                '''

        else:
            if y_test[i] == 1:  # FN
                stats[2] += 1
                """
                true_seg_img = skimage.io.imread(
                    os.path.join(FLAGS.eval_set_dir, os.path.splitext(os.path.basename(img_path))[0] + '_true_seg.png'))
                true_seg_img = true_seg_img.astype(np.float64)
                np.divide(true_seg_img, 255.0, out=true_seg_img)
                true_pixel_area = np.sum(true_seg_img)
                true_pixel_area = true_pixel_area * (RESOLUTION * RESOLUTION) / (
                            ORIGINAL_IMAGE_SIZE * ORIGINAL_IMAGE_SIZE)
                true_total_area += true_pixel_area
                """

    # report precision and recall and absolute error rate.
    abs_error_sum_r = 0
    for e in area_error:
        print('Area error: ' + str(e))
        abs_error_sum_r += abs(e)
    abs_error_rate_r = float(abs_error_sum_r) / (float(len(area_error)) + 0.00000001)

    precision_r = float(stats[0]) / float(stats[0] + stats[1] + 0.00000001)
    recall_r = float(stats[0]) / float(stats[0] + stats[2] + + 0.00000001)

    print('############ RESULTS ############')
    print('Precision: ' + str(precision_r) + '\nRecall: ' + str(recall_r) +
          '\nAverage absolute error rate: ' + str(abs_error_rate_r))

def run():
    # load model
    if args.build_model:
        model, parallel_model = build_model()
    else:
        # load old model
        model = load_model(os.path.join(LOAD_DIR, args.ckpt_load), compile=False)
        try:
            parallel_model = utils.multi_gpu_model(model)
            print("Using multithreading")
        except Exception:
            parallel_model = model
            print("Using multithreading failed")
            pass

        parallel_model.compile(optimizer='rmsprop',
                               loss='sparse_categorical_crossentropy',
                               metrics=['sparse_categorical_accuracy'])

    # load and split data
    x_train, y_train, x_test, y_test = load_data(shuffle=True)

    # fit model
    if not args.skip_train:
        parallel_model = fit(parallel_model, x_train, y_train)
        model.save(os.path.join(SAVE_DIR, f"keras_model_seg_trained_{args.fine_tune_layers}.h5"))
    elif args.verbose:
        print("Skipping training")

    # eval model
    if not args.skip_test:
        test(parallel_model, x_test, y_test)
    elif args.verbose:
        print("Skipping testing")

if __name__ == '__main__':
    args = parse_args()
    run()
