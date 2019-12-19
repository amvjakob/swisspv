import os.path
import tensorflow as tf

from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_save_dir', 'ckpt/deepsolar_classification',
                           """Directory in which to save model checkpoint. """)

tf.app.flags.DEFINE_string('ckpt_restore_dir', 'ckpt/inception_classification',
                           """Directory for restoring old model checkpoint. """)

IMAGE_SIZE = 299
BATCH_SIZE = 100
NUM_CLASSES = 2

def transform():

    # build the tensorflow graph.
    with tf.Graph().as_default() as g:

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps = 1

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(0.001,
                                        global_step,
                                        decay_steps,
                                        0.9997,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, 0.5)

        images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = inception.inference(images, NUM_CLASSES, for_training=True,
                                     restore_logits=True,
                                     scope=None)

        inception.loss(logits, labels, batch_size=BATCH_SIZE)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope=None)

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

        # remove "tower" prescript
        name_to_var_map = {var.op.name: var for var in tf.global_variables()}
        prefix = "tower_0/tower_0/"

        for faulty_var in ["CrossEntropyLoss/value/avg", "aux_loss/value/avg",
                           "total_loss/avg"]:
            name_to_var_map[prefix + faulty_var] = name_to_var_map[faulty_var]
            del name_to_var_map[faulty_var]

        # Create a saver.
        saver = tf.train.Saver(name_to_var_map)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # open session and initialize
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)

        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_restore_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        checkpoint_path = os.path.join(FLAGS.ckpt_save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=0)
        return

if __name__ == '__main__':
    transform()
