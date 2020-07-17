from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import random
import tensorflow as tf
import numpy as np
from modules.utils import load_yaml

flags.DEFINE_string('cfg_path', '../../configs/retinaface_mbv2_local.yaml',
                    'config file path')
flags.DEFINE_string('output_path', './CelebA_train_bin_100.tfrecord',
                    'path to ouput tfrecord')
flags.DEFINE_boolean('is_binary', True, 'whether save images as binary files'
                                        ' or load them on the fly.')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_example(img_name, img_path, target, is_binary):
    # Create a dictionary with features that may be relevant.
    feature = {'image/img_name': _bytes_feature([img_name]),
               'image/object/bbox/xmin': _float_feature(target[:, 0]),
               'image/object/bbox/ymin': _float_feature(target[:, 1]),
               'image/object/bbox/xmax': _float_feature(target[:, 2]),
               'image/object/bbox/ymax': _float_feature(target[:, 3]),
               'image/object/smile/valid': _float_feature(target[:, 4])}
    if is_binary:
        img_str = open(img_path, 'rb').read()
        feature['image/encoded'] = _bytes_feature([img_str])
    else:
        feature['image/img_path'] = _bytes_feature([img_path])

    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_info(dataset_path, txt_path):
    """load info from txt"""
    img_paths = []
    words = []
    f = open(txt_path, 'r')
    lines = f.readlines()[2:]
    for line in lines:
        line = line.strip().split()
        label = [float(x) for x in line[1:]]
        words.append([label])

        img_path = dataset_path + line[0]
        img_paths.append(img_path)

    return img_paths, words


def get_target(labels):
    annotations = np.zeros((0, 5))
    if len(labels) == 0:
        return annotations
    for idx, label in enumerate(labels):
        annotation = np.zeros((1, 5))
        # bbox
        annotation[0, 0] = label[0]  # x1
        annotation[0, 1] = label[1]  # y1
        annotation[0, 2] = label[0] + label[2]  # x2
        annotation[0, 3] = label[1] + label[3]  # y2

        # smiling
        annotation[0, 4] = label[4]
        annotations = np.append(annotations, annotation, axis=0)
    target = np.array(annotations)
    return target


def main(_):
    cfg = load_yaml(FLAGS.cfg_path)
    dataset_path = cfg['train_image_dataset_path']

    txt_path = './img_box_smile_100.txt'
    logging.info('Reading data list...')
    img_paths, words = load_info(dataset_path, txt_path)
    samples = list(zip(img_paths, words))
    random.shuffle(samples)

    if os.path.exists(FLAGS.output_path):
        logging.info('{:s} already exists. Exit...'.format(
            FLAGS.output_path))
        exit()

    logging.info('Writing {} sample to tfrecord file...'.format(len(samples)))
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_path, word in tqdm.tqdm(samples):
            target = get_target(word)
            img_name = os.path.basename(img_path).replace('.jpg', '')

            tf_example = make_example(img_name=str.encode(img_name),
                                      img_path=str.encode(img_path),
                                      target=target,
                                      is_binary=FLAGS.is_binary)

            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
