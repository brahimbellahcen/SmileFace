import tensorflow as tf


def _smooth_l1_loss(y_true, y_pred):
    t = tf.abs(y_pred - y_true)
    return tf.where(t < 1, 0.5 * t ** 2, t - 0.5)


def MultiBoxLoss(num_class=2, neg_pos_ratio=3):
    """multi-box loss"""

    def multi_box_loss(y_true, y_pred):
        num_batch = tf.shape(y_true)[0]
        num_prior = tf.shape(y_true)[1]

        loc_pred = tf.reshape(y_pred[0], [num_batch * num_prior, 4])
        smile_pred = tf.reshape(y_pred[1], [num_batch * num_prior, num_class])
        class_pred = tf.reshape(y_pred[2], [num_batch * num_prior, num_class])

        loc_true = tf.reshape(y_true[..., :4], [num_batch * num_prior, 4])
        smile_true = tf.reshape(y_true[..., 4], [num_batch * num_prior, 1])
        class_true = tf.reshape(y_true[..., 5], [num_batch * num_prior, 1])

        # define filter mask: class_true = 1 (pos), 0 (neg), -1 (ignore)
        mask_pos = tf.equal(class_true, 1)
        mask_neg = tf.equal(class_true, 0)

        # localization loss (smooth L1)
        mask_pos_box = tf.broadcast_to(mask_pos, tf.shape(loc_true))
        loss_loc = _smooth_l1_loss(tf.boolean_mask(loc_true, mask_pos_box),
                                   tf.boolean_mask(loc_pred, mask_pos_box))
        loss_loc = tf.reduce_mean(loss_loc)

        # smile classification loss (cross-entropy)
        mask_pos_smile = tf.broadcast_to(mask_pos, tf.shape(smile_true))
        smile_y_true = tf.boolean_mask(smile_true, mask_pos_smile)

        mask_pos_smile_b = tf.broadcast_to(mask_pos, tf.shape(smile_pred))
        smile_y_pred = tf.boolean_mask(smile_pred, mask_pos_smile_b)
        smile_y_pred = tf.reshape(smile_y_pred, [-1, num_class])

        loss_smile = tf.keras.losses.sparse_categorical_crossentropy(smile_y_true, smile_y_pred)
        loss_smile = tf.reduce_mean(loss_smile)

        # face classification loss (cross-entropy)
        # 1. compute max conf across batch for hard negative mining
        loss_class = tf.where(mask_neg,
                              1 - class_pred[:, 0][..., tf.newaxis], 0)

        # 2. hard negative mining
        loss_class = tf.reshape(loss_class, [num_batch, num_prior])
        loss_class_idx = tf.argsort(loss_class, axis=1, direction='DESCENDING')
        loss_class_idx_rank = tf.argsort(loss_class_idx, axis=1)
        mask_pos_per_batch = tf.reshape(mask_pos, [num_batch, num_prior])
        num_pos_per_batch = tf.reduce_sum(
            tf.cast(mask_pos_per_batch, tf.float32), 1, keepdims=True)
        num_pos_per_batch = tf.maximum(num_pos_per_batch, 1)
        num_neg_per_batch = tf.minimum(neg_pos_ratio * num_pos_per_batch,
                                       tf.cast(num_prior, tf.float32) - 1)
        mask_hard_neg = tf.reshape(
            tf.cast(loss_class_idx_rank, tf.float32) < num_neg_per_batch,
            [num_batch * num_prior, 1])

        # 3. classification loss including positive and negative examples
        loss_class_mask = tf.logical_or(mask_pos, mask_hard_neg)
        filter_class_true = tf.boolean_mask(tf.cast(mask_pos, tf.float32), loss_class_mask)

        loss_class_mask_b = tf.broadcast_to(loss_class_mask, tf.shape(class_pred))
        filter_class_pred = tf.boolean_mask(class_pred, loss_class_mask_b)
        filter_class_pred = tf.reshape(filter_class_pred, [-1, num_class])

        loss_class = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=filter_class_true, y_pred=filter_class_pred)
        loss_class = tf.reduce_mean(loss_class)

        return loss_loc, loss_smile, loss_class

    return multi_box_loss


# def MultiBoxLoss(num_class=1, neg_pos_ratio=3):
#     """multi-box loss"""
#
#     def multi_box_loss(y_true, y_pred):
#         num_batch = tf.shape(y_true)[0]
#         num_prior = tf.shape(y_true)[1]
#
#         loc_pred = tf.reshape(y_pred[0], [num_batch * num_prior, 4])
#         smile_pred = tf.reshape(y_pred[1], [num_batch * num_prior, 1])
#         face_pred = tf.reshape(y_pred[2], [num_batch * num_prior, 1])
#         loc_true = tf.reshape(y_true[..., :4], [num_batch * num_prior, 4])
#         smile_true = tf.reshape(y_true[..., 4], [num_batch * num_prior, 1])
#         face_true = tf.reshape(y_true[..., 5], [num_batch * num_prior, 1])
#
#         # define filter mask: face_true = 1 (pos), 0 (neg), -1 (ignore)
#         mask_pos = tf.equal(face_true, 1)
#         mask_neg = tf.equal(face_true, 0)
#
#         # localization loss (smooth L1)
#         mask_pos_b = tf.broadcast_to(mask_pos, tf.shape(loc_true))
#         loss_loc = _smooth_l1_loss(tf.boolean_mask(loc_true, mask_pos_b),
#                                    tf.boolean_mask(loc_pred, mask_pos_b))
#         loss_loc = tf.reduce_mean(loss_loc)
#
#         # smile classification loss (cross-entropy)
#         mask_pos_smile = tf.broadcast_to(mask_pos, tf.shape(smile_true))
#         loss_smile = tf.keras.losses.binary_crossentropy(
#             tf.boolean_mask(smile_true, mask_pos_smile),
#             tf.boolean_mask(smile_pred, mask_pos_smile))
#         loss_smile = tf.reduce_mean(loss_smile)
#
#         # face classification loss (cross-entropy)
#         # 1. compute max conf across batch for hard negative mining
#         loss_face = tf.where(mask_neg, 1 - face_pred[:, 0][..., tf.newaxis], 0)
#
#         # 2. hard negative mining
#         loss_face = tf.reshape(loss_face, [num_batch, num_prior])
#         loss_face_idx = tf.argsort(loss_face, axis=1, direction='DESCENDING')
#         loss_face_idx_rank = tf.argsort(loss_face_idx, axis=1)
#         mask_pos_per_batch = tf.reshape(mask_pos, [num_batch, num_prior])
#         num_pos_per_batch = tf.reduce_sum(
#             tf.cast(mask_pos_per_batch, tf.float32), 1, keepdims=True)
#         num_pos_per_batch = tf.maximum(num_pos_per_batch, 1)
#         num_neg_per_batch = tf.minimum(neg_pos_ratio * num_pos_per_batch,
#                                        tf.cast(num_prior, tf.float32) - 1)
#         mask_hard_neg = tf.reshape(
#             tf.cast(loss_face_idx_rank, tf.float32) < num_neg_per_batch,
#             [num_batch * num_prior, 1])
#
#         # 3. classification loss including positive and negative examples
#         loss_face_mask = tf.logical_or(mask_pos, mask_hard_neg)
#         loss_face_mask_b = tf.broadcast_to(loss_face_mask,
#                                            tf.shape(face_pred))
#         filter_face_true = tf.boolean_mask(tf.cast(mask_pos, tf.float32),
#                                            loss_face_mask)
#         filter_face_pred = tf.boolean_mask(face_pred, loss_face_mask_b)
#         filter_face_pred = tf.reshape(filter_face_pred, [-1, num_class])
#         loss_face = tf.keras.losses.binary_crossentropy(filter_face_true, filter_face_pred)
#         loss_face = tf.reduce_mean(loss_face)
#
#         return loss_loc, loss_smile, loss_face
#
#     return multi_box_loss

if __name__ == '__main__':
    # class_true = [0, 1, 1, 1]
    # smile_true = [0, 0, 1, 1]
    # smile_pred = [[0, 0], [0, 0], [0.1, 0.9], [0.2, 0.8]]
    # mask_pos = tf.equal(class_true, 1)
    #
    # mask_pos_smile = tf.broadcast_to(mask_pos, tf.shape(smile_true))
    # y_true = tf.boolean_mask(smile_true, mask_pos_smile)
    # y_pred = tf.boolean_mask(smile_pred, mask_pos_smile)
    # loss_smile = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # loss_smile = tf.reduce_mean(loss_smile)
    # print(loss_smile)
    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    print(loss.numpy())
