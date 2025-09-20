
import numpy as np
import random
import tensorflow as tf
import importlib.util
import cv2
import gc
# ===  Loss Functions ===
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# === Compute Shape Prior ===
def compute_shape_prior_from_generator(generator):
    all_masks = []
    for i in range(len(generator)):
        _, masks = generator[i]
        all_masks.append(masks)
    all_masks = np.concatenate(all_masks, axis=0)
    prior = np.mean(all_masks, axis=0)  # shape: (H, W, 1)
    return prior


# === Custom Total Loss ===
def total_loss_fn(prior):
    prior_tensor = tf.convert_to_tensor(prior, dtype=tf.float32)
    def loss(y_true, y_pred):
        d_loss = dice_loss(y_true, y_pred)
        s_loss = tf.reduce_mean(tf.square(y_pred - prior_tensor))
        return d_loss + 0.1 * s_loss
    return loss