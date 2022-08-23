from imports import tf

# Loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_metric(y_true, y_pred):
	return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
