"""Train the inception-v3 model on Solar Panel Identification dataset."""

from datetime import datetime
import os.path
import time
import sys

import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import random
import pickle
from collections import deque

from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_save_dir', 'ckpt/inception_classification',
                           """Directory in which to save old model checkpoint. """)

tf.app.flags.DEFINE_string('ckpt_dir_gpus', 'ckpt/inception_classification_gpus',
                           """Directory in which to save old model checkpoint when using GPUs. """)

tf.app.flags.DEFINE_string('ckpt_restore_dir', 'ckpt/inception_classification_restore',
                           """Directory for restoring old model checkpoint. """)

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 'ckpt/inception-v3/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_integer('max_steps_train', 1000,
                            """Number of training steps""")

tf.app.flags.DEFINE_integer('batch_size_train', 100,
                            """Number of samples per batch for training""")

tf.app.flags.DEFINE_integer('train_checkpoint_steps', 100,
                            """Number of steps after which to save a checkpoint.""")

tf.app.flags.DEFINE_boolean('fine_tune', True,
                            """If true, start from well-trained model on SPI dataset, else start from
                            pretrained model on ImageNet""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_boolean('skip_train', False,
                            """If true, the training of the model will be skipped.""")

tf.app.flags.DEFINE_boolean('use_gpus', False,
                            """If true, the training of the model will be done on several GPUs.""")

# basic parameters
IMAGE_SIZE = 299
NUM_CLASSES = 2

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 0.1              # Epsilon term for RMSProp.

def rotate_image(image):
    rotate_angle_list = [0, 90, 180, 270]
    rotate_angle = random.choice(rotate_angle_list)
    return skimage.transform.rotate(image, rotate_angle)

def load_image(path):
    # load image and prepocess.
    rotate_angle_list = [0, 90, 180, 270]
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True, mode='constant')
    if resized_img.shape[2] != 3:
        resized_img = resized_img[:, :, 0:3]
    rotate_angle = random.choice(rotate_angle_list)
    image = skimage.transform.rotate(resized_img, rotate_angle)
    return image

def train_input_generator(x_train, y_train, batch_size=100):
    assert len(x_train) == len(y_train)
    assert len(x_train) >= batch_size

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        x_train = np.array([ rotate_image(x) for x in x_train ])
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size

def train():
    # skip training if user wishes to do so
    if FLAGS.skip_train:
        print("Training skipped")
        return

    """
    # load train set list and transform it to queue.
    try:
        with open('train_set_list.pickle', 'rb') as f:
            train_set_list = pickle.load(f)
    except:
        raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
    """

    train_set_list = []
    train_set_size = len(train_set_list)
    x_train = [ load_image(el[0]) for el in train_set_list ]
    y_train = [ el[1] for el in train_set_list ]
    training_batch_generator = train_input_generator(x_train,
                                                     y_train,
                                                     batch_size=FLAGS.batch_size_train)
    print('Training set loaded. Size: '+str(train_set_size))

    # init horovod
    if FLAGS.use_gpus:
        import horovod.tensorflow as hvd

        # Horovod: initialize Horovod.
        hvd.init()
        assert hvd.mpi_threads_supported()

        from mpi4py import MPI
        assert hvd.size() == MPI.COMM_WORLD.Get_size()

    # build the tensorflow graph.
    with tf.Graph().as_default() as g:

        # global_step = slim.variables.global_step()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.zeros_initializer,
            trainable=False)

        """global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False)"""

        num_batches_per_epoch = train_set_size / FLAGS.batch_size_train
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        lr_scaler = 1
        """if FLAGS.use_gpus:
            lr_scaler = hvd.size()"""

        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr * lr_scaler, RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)

        if FLAGS.use_gpus:
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt, op=hvd.Average)

        images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_train, IMAGE_SIZE, IMAGE_SIZE, 3])

        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size_train])

        logits = inception.inference(images, NUM_CLASSES, for_training=True,
                                     restore_logits=FLAGS.fine_tune,
                                     scope=None)

        inception.loss(logits, labels, batch_size=FLAGS.batch_size_train)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope = None)

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope=None)

        # Calculate the gradients for the batch of data on this ImageNet
        # tower.
        grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY, global_step)

        variables_to_average = (tf.trainable_variables() +
                                tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        # remove "tower" prescript
        name_to_var_map = { var.op.name: var for var in tf.global_variables() }
        prefix = "tower_0/tower_0/"

        for faulty_var in [ "CrossEntropyLoss/value/avg", "aux_loss/value/avg",
                            "total_loss/avg" ]:
            name_to_var_map[prefix + faulty_var] = name_to_var_map[faulty_var]
            del name_to_var_map[faulty_var]

        saver = tf.train.Saver(name_to_var_map)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Horovod: set hooks
        hooks = []
        if FLAGS.use_gpus:
            hooks = [
                # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
                # from rank 0 to all other processes. This is necessary to ensure consistent
                # initialization of all workers when training is started with random weights
                # or restored from a checkpoint.towchr
                hvd.BroadcastGlobalVariablesHook(0)
            ]

        # open session and initialize
        config = tf.ConfigProto(log_device_placement=False)
        if FLAGS.use_gpus:
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())

        sess = tf.Session(config=config)
        sess.run(init)

        summary_writer = tf.summary.FileWriter(
            FLAGS.ckpt_save_dir,
            graph_def=sess.graph.as_graph_def(add_shapes=True))

        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        global_step = tf.cast(global_step, tf.int32)
        print(global_step.dtype)

        checkpoint_path = os.path.join(FLAGS.ckpt_save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=0)
        return

        step = 1
        while step <= FLAGS.max_steps_train:
            start_time = time.time()

            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)
            image_batch = np.reshape(image_, [FLAGS.batch_size_train, IMAGE_SIZE, IMAGE_SIZE, 3])
            label_batch = np.reshape(label_, [FLAGS.batch_size_train])

            _, loss_value = sess.run([train_op, total_loss], feed_dict={images: image_batch, labels: label_batch})

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step == 1 or step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size_train
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')

                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # write summary periodically
            if step == 1 or step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: image_batch, labels: label_batch})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.train_checkpoint_steps == 0 or step == 2:
                checkpoint_path = os.path.join(FLAGS.ckpt_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1

if __name__ == '__main__':
    train()
