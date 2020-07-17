from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from modules.anchor import prior_box_tf, decode_tf
from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2_remote.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '-1', 'which gpu to use')
flags.DEFINE_string('img_path', '/Users/lichaochao/Downloads/CelebA/CelebA/test/200001.jpg', 'path to input image')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.1, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.9, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 0.5, 'down-scale factor for inputs')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load model from weights.h5
    # model.load_weights('./model/mbv2_weights.h5', by_name=True, skip_mismatch=True)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    if not FLAGS.webcam:
        file_path = '/Users/lichaochao/Downloads/images_UMU/'
        for file_name in os.listdir(file_path + 'source_images/'):
            image_path = file_path + 'source_images/' + file_name
            if not os.path.exists(image_path):
                print(f"cannot find image path from {image_path}")
                continue

            img_raw = cv2.imread(image_path)
            img = np.float32(img_raw.copy())

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
            img_height, img_width, _ch = img.shape

            # run model
            outputs = model(img[np.newaxis, ...])

            preds = tf.concat(
                [outputs[0][0], outputs[1][0, :, 1][..., tf.newaxis],
                 outputs[2][0, :, 1][..., tf.newaxis]], -1)

            priors = prior_box_tf((img_height, img_width),
                                  cfg['min_sizes'], cfg['steps'], cfg['clip'])
            decode_preds = decode_tf(preds, priors, cfg['variances'])

            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=FLAGS.iou_th,
                score_threshold=FLAGS.score_th)

            outputs = tf.gather(decode_preds, selected_indices).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)
            has_face = False
            is_smile = False
            for prior_index in range(len(outputs)):
                ann = outputs[prior_index]
                if ann[-1] >= 0.5:
                    has_face = True
                    x1, y1 = int(ann[0] * img_width), int(ann[1] * img_height)
                    x2, y2 = int(ann[2] * img_width), int(ann[3] * img_height)

                    text = "face: {:.2f}".format(ann[-1] * 100)
                    cv2.putText(img, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    if ann[-2] >= 0.5:
                        is_smile = True
                        smile_text = "smile: {:.2f}".format(ann[-2] * 100)
                        cv2.putText(img, smile_text, (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if is_smile:
                dst_file_path = file_path + '/smile_face/' + file_name
            elif has_face:
                dst_file_path = file_path + '/face/' + file_name
            else:
                dst_file_path = file_path + '/no_face/' + file_name
            cv2.imwrite(dst_file_path, img)
            print(dst_file_path)

    else:
        cam = cv2.VideoCapture('./data/linda_umu.mp4')
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        resize = FLAGS.down_scale_factor
        frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize
        frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH) * resize

        max_steps = max(cfg['steps'])
        img_pad_h = max_steps - frame_height % max_steps if frame_height % max_steps > 0 else 0
        img_pad_w = max_steps - frame_width % max_steps if frame_width % max_steps > 0 else 0
        priors = prior_box_tf((frame_height + img_pad_h, frame_width + img_pad_w),
                              cfg['min_sizes'], cfg['steps'], cfg['clip'])

        frame_index = 0
        outputs = []
        start_time = time.time()
        while cam.isOpened():
            _, frame = cam.read()
            if frame is None:
                print('no cam')
                break
            if frame_index < 5:
                frame_index += 1
                # continue
            else:
                frame_index = 0

                img = np.float32(frame.copy())
                if resize < 1:
                    img = cv2.resize(img, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # pad input image to avoid unmatched shape problem
                img, pad_params = pad_input_image(img, max_steps=max_steps)

                # run model
                outputs = model(img[np.newaxis, ...])

                preds = tf.concat(
                    [outputs[0][0], outputs[1][0, :, 1][..., tf.newaxis],
                     outputs[2][0, :, 1][..., tf.newaxis]], -1)

                decode_preds = decode_tf(preds, priors, cfg['variances'])

                selected_indices = tf.image.non_max_suppression(
                    boxes=decode_preds[:, :4],
                    scores=decode_preds[:, -1],
                    max_output_size=tf.shape(decode_preds)[0],
                    iou_threshold=FLAGS.iou_th,
                    score_threshold=FLAGS.score_th)

                outputs = tf.gather(decode_preds, selected_indices).numpy()

                # recover padding effect
                outputs = recover_pad_output(outputs, pad_params, resize=resize)

                # calculate fps
                fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
                start_time = time.time()
                cv2.putText(frame, fps_str, (25, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

            # draw results
            for prior_index in range(len(outputs)):
                draw_bbox_landm(frame, outputs[prior_index], frame_height,
                                frame_width)

            # calculate fps
            # fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            # start_time = time.time()
            # cv2.putText(frame, fps_str, (25, 25),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
