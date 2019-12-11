import os.path
import argparse

import numpy as np
import tensorflow as tf

from keras.applications import InceptionV3

# constants for model loading
PATH_OLD_MODEL_DIR = os.path.join('ckpt', 'inception_classification')
PATH_OLD_MODEL_META = os.path.join(PATH_OLD_MODEL_DIR, 'model.ckpt-0.meta')
PATH_OLD_MODEL_WEIGHTS = os.path.join(PATH_OLD_MODEL_DIR, 'checkpoint')

PATH_NEW_MODEL_DIR = os.path.join('ckpt', 'inception_tl')
PATH_NEW_MODEL = os.path.join(PATH_NEW_MODEL_DIR, 'keras_swisspv_untrained.h5')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # init new inception3 model
    inception = InceptionV3()

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
        for i, var in enumerate(vars_global):
            if args.verbose > 1:
                print(f"[{i+1}/{len(vars_global)}]")
            try:
                model_vars[var.name] = var.eval()
                #print(var.name)

            except Exception as e:
                print("For var={}, an exception occurred".format(var.name))
                if args.verbose:
                    print(type(e))    # the exception instance
                    print(e.args)     # arguments stored in .args
                    print(e)

        if args.verbose:
            print("Loaded all layers.")
            print("Transforming all layers.")

        # transform layer names
        # concat all layers from old model that share the same prefix
        # into a single layer, as required by the new Keras model
        input_vars = {}
        names = []
        name = None
        value = []
        for layer in model_vars:
            prefix = layer.split('/')[0]
            if name is None:
                name = prefix

            if prefix == name:
                value.append(model_vars[layer])
            else:
                input_vars[name] = np.array(value)
                names.append(name)
                name = prefix
                value = [model_vars[layer]]
        # last layer
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
            "mixed_8x8x2048b"  : "mixed10",

        }

        # apply loaded weights to new model
        # check whether the weights within the same layer are in the same order?
        matched_layers = []
        for i, keras_layer in enumerate(inception.layers):
            if args.verbose and i < len(names):
                print(f"Trying to match layer {i}, {keras_layer.name} with {names[i]}")
            if keras_layer.name in input_vars:
                tf_layer = input_vars[keras_layer.name]
                keras_shape = np.array(keras_layer.get_weights()).shape
                # shape must match
                if tf_layer.shape == keras_shape:
                    keras_layer.set_weights(tf_layer)
                    matched_layers.append(keras_layer.name)
                else:
                    if args.verbose > 1:
                        print(f"Shape mismatch in layer {keras_layer.name} with shapes "
                              f"{tf_layer.shape} (tf) and {keras_shape} (keras)")
            elif args.verbose > 1:
                print(f"Skipping layer {keras_layer.name} with shape "
                      f"{np.array(keras_layer.get_weights()).shape}")

        if args.verbose: print("Transformed layers")
        if args.verbose > 1:
            unmatched_layers = []
            for layer in input_vars:
                if layer not in matched_layers:
                    unmatched_layers.append(layer)
                    print(layer)

            print(f"Unmatched layers from TF model: {len(unmatched_layers)}")

        # save new model
        inception.save(PATH_NEW_MODEL)
        if args.verbose: print("Saved model")
