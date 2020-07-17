from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import time

from modules.models_remote import RetinaFaceModel
from modules.lr_scheduler import MultiStepWarmUpLR
from modules.losses import MultiBoxLoss
from modules.anchor import prior_box
from modules.utils import load_yaml, load_dataset, ProgressBar

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2_local.yaml', 'config file path')


def main(_):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    strategy = tf.distribute.MirroredStrategy()

    # Global batch size
    GLOBAL_BATCH_SIZE = cfg['batch_size'] * strategy.num_replicas_in_sync

    # define prior box
    priors = prior_box((cfg['input_size'], cfg['input_size']),
                       cfg['min_sizes'], cfg['steps'], cfg['clip'])
    # load dataset
    train_dataset = load_dataset(cfg, priors, shuffle=True)
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    with strategy.scope():
        # define model
        model = RetinaFaceModel(cfg)
        model.summary()

        # define losses function
        multi_box_loss = MultiBoxLoss()

        # define optimizer
        steps_per_epoch = cfg['dataset_len'] // GLOBAL_BATCH_SIZE
        learning_rate = MultiStepWarmUpLR(
            initial_learning_rate=cfg['init_lr'],
            lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
            lr_rate=cfg['lr_rate'],
            warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
            min_lr=cfg['min_lr'])
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True)

        # load checkpoint12
        checkpoint_dir = './checkpoints/' + cfg['sub_name']
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                         optimizer=optimizer,
                                         model=model)

        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                        directory=checkpoint_dir,
                                                        max_to_keep=10)
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print('[*] load ckpt from {} at step {}.'.format(
                checkpoint_manager.latest_checkpoint, checkpoint.step.numpy()))
        else:
            model.load_weights('./model/mbv2_weights.h5', by_name=True, skip_mismatch=True)
            print("[*] training from scratch.")

    with strategy.scope():
        # define training step function
        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)

                losses = {}
                losses['reg'] = tf.reduce_sum(model.losses)
                losses['loc'], losses['smile'], losses['face'] = \
                    multi_box_loss(labels, predictions)
                total_loss = tf.add_n([l for l in losses.values()])

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return total_loss

        @tf.function
        def distributed_train_step(dataset_inputs, dataset_labels):
            per_replica_losses = strategy.experimental_run_v2(
                train_step, args=(dataset_inputs, dataset_labels)
            )
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    with strategy.scope():
        prog_bar = ProgressBar(steps_per_epoch, checkpoint.step.numpy() % steps_per_epoch)
        for inputs, labels in dist_dataset:
            checkpoint.step.assign_add(1)
            steps = checkpoint.step.numpy()
            total_loss = distributed_train_step(inputs, labels)
            prog_bar.update("epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                ((steps - 1) // steps_per_epoch) + 1, cfg['epoch'], total_loss, optimizer.lr(steps).numpy()))

            if steps % cfg['save_steps'] == 0:
                checkpoint_manager.save()
                print("\n[*] save ckpt file at {}".format(checkpoint_manager.latest_checkpoint))

        checkpoint_manager.save()
        print("\n[*] training done! save ckpt file at {}".format(checkpoint_manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)
