import tensorflow as tf

def fnc_score(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    # compute max_score = 0.25*unrelated + (agree+disagree+discuss)
    total_count = tf.cast(tf.size(y_true), dtype=tf.float32)
    unrelated_count = tf.cast(tf.math.reduce_sum(tf.cast(tf.equal(tf.constant(0, dtype=tf.int64),y_true), tf.float32)), dtype=tf.float32)
    related_count = tf.cast(tf.math.subtract(total_count, unrelated_count), dtype=tf.float32)
    max_score = tf.add(tf.math.scalar_mul(0.25, unrelated_count), related_count)

    # compute score
    unrelated_pred = tf.cast(tf.equal(tf.cast(0, dtype=tf.int64), y_pred), dtype=tf.int64)
    unrelated_true = tf.cast(tf.equal(tf.cast(0, dtype=tf.int64), y_true), dtype=tf.int64)
    correct_unrelated_count = tf.math.reduce_sum(tf.cast(tf.equal(unrelated_pred, unrelated_true), dtype=tf.float32))
    correct_unrelated_count_score = tf.math.scalar_mul(0.25, correct_unrelated_count)

    is_related_mask = tf.not_equal(tf.cast(0, dtype=tf.int64), y_pred)
    is_correct_mask = tf.equal(y_true, y_pred)

    combined_mask_correct_related = tf.logical_and(is_related_mask, is_correct_mask)
    correct_related_count = tf.math.reduce_sum(tf.cast(combined_mask_correct_related, dtype=tf.float32))
    correct_related_count_score = tf.math.scalar_mul(1.0, correct_related_count)

    is_related_true_mask = tf.not_equal(tf.cast(0, dtype=tf.int64), y_true)
    combined_mask_related = tf.logical_and(is_related_mask, is_related_true_mask)
    combined_mask_incorrect_related = tf.logical_and(combined_mask_related, tf.logical_not(combined_mask_correct_related))
    incorrect_related_count = tf.math.reduce_sum(tf.cast(combined_mask_incorrect_related, dtype=tf.float32))
    incorrect_related_count_score = tf.math.scalar_mul(0.25, incorrect_related_count)

    score = tf.math.add_n([correct_unrelated_count_score, correct_related_count_score, incorrect_related_count_score])
    return tf.math.divide(score, max_score)
    