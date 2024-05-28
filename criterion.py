import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend

class BasnetLoss(keras.losses.Loss):
    """BASNet hybrid loss."""

    def __init__(self, reduction=keras.losses.Reduction.AUTO, name="basnet_loss"):
        super().__init__(reduction=reduction, name=name)
        self.smooth = 1.0e-9

        # Sparse Categorical Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        # Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(self, y_true, y_pred):
        """Calculate intersection over union (IoU) between images."""
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        intersection = backend.sum(backend.abs(y_true * y_pred), axis=[1, 2, 3])
        union = backend.sum(y_true, [1, 2, 3]) + backend.sum(y_pred, [1, 2, 3])
        union = union - intersection

        return backend.mean((intersection + self.smooth) / (union + self.smooth), axis=0)

    def call(self, y_true, y_pred):
        cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred)

        # Convert logits to class predictions
        y_pred_class = tf.argmax(y_pred, axis=-1)
        y_pred_class = tf.expand_dims(y_pred_class, axis=-1)

        # Convert to float32 for SSIM calculation
        y_pred_class = tf.cast(y_pred_class, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        ssim_value = self.ssim_value(y_true, y_pred_class, max_val=1.0)
        ssim_loss = backend.mean(1 - ssim_value + self.smooth, axis=0)

        iou_value = self.iou_value(y_true, y_pred_class)
        iou_loss = 1 - iou_value

        # Add all three losses.
        return cross_entropy_loss + ssim_loss + iou_loss
