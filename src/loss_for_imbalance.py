import tensorflow as tf
tf.set_random_seed(2016)
from keras import backend as K

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))
  