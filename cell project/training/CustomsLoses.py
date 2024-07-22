import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
'''
implementaciones otras loses para keras
'''


@tf.keras.utils.register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

# Register the custom loss function
get_custom_objects().update({"dice_coef_loss": dice_coef_loss})


@tf.keras.utils.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.5, beta=2):
    # Convert predictions to probabilities
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    tp = K.sum(y_true * y_pred, axis=-1)
    fp = K.sum((1 - y_true) * y_pred, axis=-1)
    fn = K.sum(y_true * (1 - y_pred), axis=-1)

    # Calculate the Tversky index
    tversky_index = (tp + K.epsilon()) / (tp + alpha * fp + beta * fn + K.epsilon())

    # Return the Tversky loss
    return 1 - tversky_index
get_custom_objects().update({"tversky_loss": tversky_loss})


@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1,2,3])
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return iou
get_custom_objects().update({"iou": iou})

@tf.keras.utils.register_keras_serializable()
def jaccard_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (intersection + 1) / (union + 1)
get_custom_objects().update({"jaccard_loss": jaccard_loss})

@tf.keras.utils.register_keras_serializable()
def pixel_accuracy(y_true, y_pred):
    y_pred = K.round(y_pred)
    correct_pixels = K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()))
    total_pixels = K.cast(K.prod(K.shape(y_true)), K.floatx())
    return (correct_pixels / total_pixels)
get_custom_objects().update({"pixel_accuracy": pixel_accuracy})


@tf.keras.utils.register_keras_serializable()
def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(tf.round(y_pred), tf.int32)

    # True Negatives
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32))

    # False Positives
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))

    specificity = tn / (tn + fp + K.epsilon())
    return specificity
get_custom_objects().update({"specificity": specificity})

@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred):
        gamma=2.,
        alpha=0.25
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Clip the prediction values to prevent NaNs
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Compute cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred)

        # Compute weight
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)

        # Compute Focal Loss
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)

get_custom_objects().update({"focal_loss": focal_loss})

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    # Binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # Dice loss
    dice = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

    return bce + dice