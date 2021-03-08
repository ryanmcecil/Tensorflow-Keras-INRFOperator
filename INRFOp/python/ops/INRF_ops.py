"""Ryan Cecil. Duquesne University 2020. INRF2d Operator and Register Gradient"""

from tensorflow.python.framework import ops
import tensorflow as tf

def tf_pad_grad(grad,w_size):
    pd = int((w_size[0] - 1))
    return tf.pad(grad,[[0,0],[pd,pd],[pd,pd],[0,0]],'CONSTANT')

def tf_rot180(w):
    return tf.reverse(w, axis=[0, 1])

def permute_rot(w):
    return tf.transpose(tf_rot180(w),perm=[0,1,3,2])

def tf_NHWC_to_HWIO(out):
    """
    Converts [batch, in_height, in_width, in_channels]
    to       [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.transpose(out, perm=[1, 2, 0, 3])

#Must be modified to match system
INRF_module0 = tf.load_op_library(
        '{path to tensorflow}/tensorflow-r2.3/bazel-bin/tensorflow/core/user_ops/INRF2d_gpu.so')


def INRF2d(x,m,w,g,lambdaa):
    return INRF_module0.INRF2d(x=x,m=m,w=w,g=g,lamda = lambdaa)


@ops.RegisterGradient("INRF2d")
def _INRF2dGrad(op, grad):
    x_grad = INRF_module0.INRF2dGradX(x = op.inputs[0], m = op.inputs[1], w = op.inputs[2], g = op.inputs[3], lamda = op.inputs[4], grad = grad)
    x_grad = tf.nn.conv2d(tf_pad_grad(grad,op.inputs[2].shape), permute_rot(op.inputs[1]), strides = [1,1,1,1], padding = 'VALID') + x_grad
    m_grad = tf_NHWC_to_HWIO(tf.nn.conv2d(tf.transpose(op.inputs[0],perm=[3,1,2,0]),
                                          tf_NHWC_to_HWIO(grad),
                                          strides=[1,1,1,1],
                                          padding='VALID'))
    w_grad = INRF_module0.INRF2dGradW(x = op.inputs[0], g = op.inputs[3], lamda = op.inputs[4], grad = grad)
    g_grad = INRF_module0.INRF2dGradG(x = op.inputs[0], w = op.inputs[2], g = op.inputs[3], lamda = op.inputs[4], grad = grad)
    lambdaa_grad = INRF_module0.INRF2dGradL(x = op.inputs[0], w=op.inputs[2], g=op.inputs[3], grad=grad)
    return [x_grad, m_grad, w_grad, g_grad, lambdaa_grad]
