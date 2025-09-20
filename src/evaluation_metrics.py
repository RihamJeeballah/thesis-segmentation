import numpy as np
def dice_score(y_true, y_pred, smooth=1e-6):
    inter = np.sum(y_true * y_pred)
    return (2.0 * inter) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)