from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import RetinaFaceModel
from modules.lr_scheduler import MultiStepWarmUpLR
from modules.losses import MultiBoxLoss
from modules.anchor import prior_box
from modules.utils import (set_memory_growth, load_yaml, load_dataset,
                           ProgressBar)

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2_320_local.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '-1', 'which gpu to use')


def main(_):
    # init
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=True)
    model.summary()

    # define prior box
    priors = prior_box((cfg['input_size'], cfg['input_size']),
                       cfg['min_sizes'], cfg['steps'], cfg['clip'])

    # load dataset
    train_dataset = load_dataset(cfg, priors, shuffle=True)

    # define optimizer
    steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']
    learning_rate = MultiStepWarmUpLR(
        initial_learning_rate=cfg['init_lr'],
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
        lr_rate=cfg['lr_rate'],
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
        min_lr=cfg['min_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)

    # define losses function
    multi_box_loss = MultiBoxLoss()

    # load checkpoint12
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)

    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        model.load_weights('./model/mbv2_weights.h5', by_name=True, skip_mismatch=True)
        print("[*] training from {}.")

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

    # training loop
    remain_steps = max(steps_per_epoch * cfg['epoch'] - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch, checkpoint.step.numpy() % steps_per_epoch)

    for inputs, labels in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss = train_step(inputs, labels)

        prog_bar.update("epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
            ((steps - 1) // steps_per_epoch) + 1, cfg['epoch'],
            total_loss, optimizer.lr(steps).numpy()))

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))

    manager.save()
    print("\n[*] training done! save ckpt file at {}".format(
        manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)
