import os.path
import argparse

import numpy as np
import tensorflow as tf

from keras.applications import InceptionV3
from keras.layers.advanced_activations import Softmax
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import AveragePooling2D, Conv2D
from keras import initializers
from keras.models import Model

from keras import backend as K

# constants for model loading and saving
PATH_OLD_MODEL_DIR = os.path.join('ckpt', 'deepsolar_classification')
PATH_OLD_MODEL_META = os.path.join(PATH_OLD_MODEL_DIR, 'model.ckpt-0.meta')
PATH_OLD_MODEL_WEIGHTS = os.path.join(PATH_OLD_MODEL_DIR, 'checkpoint')

PATH_NEW_MODEL_DIR = os.path.join('ckpt', 'inception_tl_load')

NUM_CLASSES = 2

def parse_args():
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
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--with_aux', type=str2bool, nargs='?',
                        const=True, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # init new inception3 model
    inception = InceptionV3(include_top=False,
                            input_shape=(299, 299, 3),
                            weights=None,
                            pooling=None)
    output = inception.layers[-1].output

    # add last layers
    net = AveragePooling2D(pool_size=(8, 8), # K.int_shape(output),
                                  strides=(2, 2),
                                  padding='valid',
                                  data_format='channels_last',
                                  name='pool')(output)
    net = Dropout(rate=0.2,
                         name='dropout')(net)
    net = Flatten(data_format='channels_last',
                         name='flatten')(net)

    logits = Dense(NUM_CLASSES, activation=None, name="logits")(net)
    predictions = Softmax(name='predictions')(logits)

    # aux net
    if args.with_aux:
        output = inception.get_layer(name="mixed7").output
        aux_logits = AveragePooling2D(pool_size=(5, 5),
                                      strides=(3, 3),
                                      padding='valid')(output)
        aux_logits = Conv2D(128, (1, 1), name='proj')(aux_logits)
        aux_logits = Conv2D(768, K.int_shape(aux_logits)[1:3],
                            kernel_initializer=initializers.TruncatedNormal(stddev=0.001),
                            padding='valid')(aux_logits)
        aux_logits = Flatten(data_format='channels_last')(aux_logits)
        aux_logits = Dense(NUM_CLASSES, activation=None, name='aux_logits',
                           kernel_initializer=initializers.TruncatedNormal(stddev=0.001))(aux_logits)
        aux_predictions = Softmax(name='aux_predictions')(aux_logits)

        model = Model(inputs=inception.input, outputs=[predictions, aux_predictions])
        model.save(os.path.join(PATH_NEW_MODEL_DIR, 'keras_inception_untrained_aux.h5'))
    else:
        model = Model(inputs=inception.input, outputs=predictions)
        model.save(os.path.join(PATH_NEW_MODEL_DIR, 'keras_inception_untrained.h5'))

    # start tensorflow session
    with tf.Session() as sess:

        # import graph
        saver = tf.train.import_meta_graph(PATH_OLD_MODEL_META)
        init = tf.global_variables_initializer()
        sess.run(init)

        # load weights for graph
        checkpoint = tf.train.get_checkpoint_state(PATH_OLD_MODEL_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            raise Exception("Checkpoint not found")

        # get all global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        n_vars = 0
        for i, var in enumerate(vars_global):
            if args.verbose > 1:
                print(f"[{i+1}/{len(vars_global)}]")

            try:
                val = var.eval()
                model_vars[var.name] = val
                n_vars += val.size

            except Exception as e:
                print("For var={}, an exception occurred".format(var.name))
                if args.verbose:
                    print(type(e))
                    print(e.args)
                    print(e)

        if args.verbose:
            print("Loaded all layers.")
            print(f"There are a total of {n_vars} weights")
            print("Transforming all layers.")

        # transform layer names
        # concat all layers from old model that share the same prefix
        # into a single layer, as required by the new Keras model
        input_vars = {}
        names = []
        name = None
        value = []
        batch_norms = []
        for layer in model_vars:
            prefix = layer.split('/')[0]
            if name is None:
                name = prefix

            if prefix == name:
                if prefix.startswith('batch_normalization'):
                    # only append the first three variables with this name:
                    # mean, moving_average, moving_variance
                    if len(value) < 3:
                        value.append(model_vars[layer])
                    else:
                        print(f"Trying to add more than 3 params to batch_norm layer by trying {layer}")
                else:
                    value.append(model_vars[layer])
            else:
                if name not in names:
                    input_vars[name] = np.array(value)
                    names.append(name)
                else:
                    print(f"Tried to overwrite layer {name}")

                name = prefix
                value = [model_vars[layer]]
        # last layer
        if name not in names:
            input_vars[name] = np.array(value)
            names.append(name)

        if args.verbose:
            print("Transformed all layers.")
            print("Applying weights to Keras model.")

        tf_to_keras_map = {
            "mixed_35x35x256a" : "mixed0",
            "mixed_35x35x288a" : "mixed1",
            "mixed_35x35x288b" : "mixed2",
            "mixed_17x17x768a" : "mixed3",
            "mixed_17x17x768b" : "mixed4",
            "mixed_17x17x768c" : "mixed5",
            "mixed_17x17x768d" : "mixed6",
            "mixed_17x17x768e" : "mixed7",
            "mixed_17x17x1280a": "mixed8",
            "mixed_8x8x2048a"  : "mixed9",
            "mixed_8x8x2048b"  : "mixed10"
        }

        # apply loaded weights to new model
        # check whether the weights within the same layer are in the same order?
        matched_layers = []
        n_matched = 0
        for i, keras_layer in enumerate(model.layers):
            if keras_layer.name in input_vars:
                if args.verbose:
                    print(f"Trying to match layer {i}, {keras_layer.name}")

                tf_layer = input_vars[keras_layer.name]
                keras_shape = np.array(keras_layer.get_weights()).shape
                # shape must match
                if tf_layer.shape == keras_shape:
                    keras_layer.set_weights(tf_layer)
                    matched_layers.append(keras_layer.name)
                    n_matched += np.array(keras_layer.get_weights()).size
                else:
                    if args.verbose > 1:
                        print(f"Shape mismatch in layer {keras_layer.name} with shapes "
                              f"{tf_layer.shape} (tf) and {keras_shape} (keras)")
            elif args.verbose > 1:
                print(f"Skipping layer {keras_layer.name} with shape "
                      f"{np.array(keras_layer.get_weights()).shape}")

        if args.verbose:
            print("Transformed layers")
            print(f"Applied {n_matched}/{n_vars} weights")
        if args.verbose > 1:
            unmatched_layers = []
            for layer in input_vars:
                if layer not in matched_layers:
                    unmatched_layers.append(layer)
                    print(layer)

            print(f"Unmatched layers from TF model: {len(unmatched_layers)}")

        # save new model
        if args.with_aux:
            model.save(os.path.join(PATH_NEW_MODEL_DIR, 'keras_swisspv_untrained_aux.h5'))
        else:
            model.save(os.path.join(PATH_NEW_MODEL_DIR, 'keras_swisspv_untrained.h5'))

        if args.verbose:
            print("Saved model")