"""Define losses."""

import tensorflow as tf
_EPSILON = 1e-7

def _smooth_one_hot_labels(logits, labels, label_smoothing):
  label_smoothing = tf.constant(label_smoothing, dtype=logits.dtype)
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.cast(num_classes - 1, label_smoothing.dtype),
      dtype=logits.dtype)

def _softmax_cross_entropy(logits, labels, label_smoothing, training, from_logits=True):
  # Computes the softmax in full precision.
  if logits.dtype.base_dtype != tf.float32:
    logits = tf.cast(logits, tf.float32)
  if not from_logits:
    epsilon = tf.convert_to_tensor(_EPSILON, dtype=logits.dtype.base_dtype)
    logits = tf.clip_by_value(logits, epsilon, 1.0 - epsilon)
    logits = tf.log(logits)

  if training and label_smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
      smoothed_labels = tf.stop_gradient(smoothed_labels)
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    else:
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
        logits=logits, labels=smoothed_labels)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                label_smoothing=0.0,
                                average_in_time=False,
                                mask=None,
                                mode=tf.estimator.ModeKeys.TRAIN,
                                from_logits=True,
                                training=None):
  """Computes the cross entropy loss of sequences.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The length of each sequence.
    label_smoothing: The label smoothing value.
    average_in_time: If ``True``, also average the loss in the time dimension.
    mask: [batch_size, seq_len] tf.bool or tf.int64 or tf.float32.
    mode: A ``tf.estimator.ModeKeys`` mode.
    from_logits: Boolean. whether `logits` is the result of softmax, or is the result
    of logits.
    from_logits: Boolean, whether `output` is the
        result of a softmax, or is a tensor of logits.
    training: Compute training loss. If not set, infer training mode from
      :obj:`mode`.

  Returns:
    A tuple (cumulated loss, loss normalizer, token-level normalizer).
  """
  if training is None:
    training = mode == tf.estimator.ModeKeys.TRAIN
  batch_size = tf.shape(logits)[0]
  max_time = tf.shape(logits)[1]

  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing,
                                         training, from_logits=from_logits)
  weights = tf.sequence_mask(
      sequence_length, maxlen=max_time, dtype=cross_entropy.dtype)
  if mask is not None:
    weights = weights * tf.cast(mask, dtype=cross_entropy.dtype)
  loss = tf.reduce_sum(cross_entropy * weights)
  loss_token_normalizer = tf.reduce_sum(weights)

  if average_in_time or not training:
    loss_normalizer = loss_token_normalizer
  else:
    loss_normalizer = tf.cast(batch_size, loss.dtype)

  return loss, loss_normalizer, loss_token_normalizer

def cross_entropy_loss(logits,
                       labels,
                       label_smoothing=0.0,
                       mode=tf.estimator.ModeKeys.TRAIN,
                       training=None):
  """Computes the cross entropy loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    label_smoothing: The label smoothing value.
    mode: A ``tf.estimator.ModeKeys`` mode.
    training: Compute training loss. If not set, infer training mode from
      :obj:`mode`.

  Returns:
    The cumulated loss and the loss normalizer.
  """
  if training is None:
    training = mode == tf.estimator.ModeKeys.TRAIN
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)
  loss = tf.reduce_sum(cross_entropy)
  loss_normalizer = tf.cast(tf.shape(cross_entropy)[0], loss.dtype)
  return loss, loss_normalizer
