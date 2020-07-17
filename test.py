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

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2_local.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '-1', 'which gpu to use')
flags.DEFINE_string('img_path', '/Users/lichaochao/Downloads/CelebA/CelebA/test/200001.jpg', 'path to input image')
flags.DEFINE_boolean('webcam', True, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.1, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.99, 'score threshold for nms')
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
        if not os.path.exists(FLAGS.img_path):
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

        print("[*] Processing on single image {}".format(FLAGS.img_path))

        img_raw = cv2.imread(FLAGS.img_path)
        img = np.float32(img_raw.copy())

        # testing scale
        target_size = 320
        img_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(img_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize,
                         interpolation=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # draw and save results
        save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
        for prior_index in range(len(outputs)):
            draw_bbox_landm(img, outputs[prior_index], target_size, target_size)
        cv2.imwrite(save_img_path, img)
        print(f"[*] save result at {save_img_path}")

    else:
        cam = cv2.VideoCapture('./data/lichaochao.mp4')
        # cam = cv2.VideoCapture(0)
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cam.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('chaochao1.mp4', fourcc, fps=fps, frameSize=(frame_height, frame_width))

        resize = FLAGS.down_scale_factor
        frame_height *= resize
        frame_width *= resize

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
                # fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
                # start_time = time.time()
                # cv2.putText(frame, fps_str, (25, 50),
                #             cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

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
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
