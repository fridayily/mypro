"""
This is a straightforward Python implementation of a generative adversarial network.
The code is derived from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

The tutorial's code trades efficiency for clarity in explaining how GANs function;
this script refactors a few things to improve performance, especially on GPU machines.
In particular, it uses a TensorFlow operation to generate random z values and pass them
to the generator; this way, more computations are contained entirely within the
TensorFlow graph.

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot  as plt

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Define the discriminator network
def discriminator(images, reuse_variables=None):
    # variable_scope 创建变量空间，管理由get_variable创建的命名空间
    # get_variable_scope用来重复使用当前变量空间中的变量
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features 输入1维，输出32维
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02)) #Tensor("d_w1/read:0", shape=(5, 5, 1, 32), dtype=float32)
        print('___________________')
        tf.Print(d_w1,[d_w1])
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0)) # Tensor("d_b1/read:0", shape=(32,), dtype=float32)
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME') #Tensor("Conv2D_3:0", shape=(?, 28, 28, 32), dtype=float32)
        d1 = d1 + d_b1 # Tensor("add_4:0", shape=(?, 28, 28, 32), dtype=float32) .....Tensor("add_8:0", shape=(50, 28, 28, 32), dtype=float32)
        d1 = tf.nn.relu(d1) # Tensor("Relu_3:0", shape=(?, 28, 28, 32), dtype=float32) .....Tensor("Relu_6:0", shape=(50, 28, 28, 32), dtype=float32)
        # 池化层
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Tensor("AvgPool:0", shape=(?, 14, 14, 32), dtype=float32) .....Tensor("AvgPool_2:0", shape=(50, 14, 14, 32), dtype=float32)

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02)) # Tensor("d_w2/read:0", shape=(5, 5, 32, 64), dtype=float32) ......Tensor("d_w2/read:0", shape=(5, 5, 32, 64), dtype=float32)
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0)) # Tensor("d_b2/read:0", shape=(64,), dtype=float32) ......Tensor("d_b2/read:0", shape=(64,), dtype=float32)
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME') # Tensor("Conv2D_4:0", shape=(?, 14, 14, 64), dtype=float32) .....Tensor("Conv2D_6:0", shape=(50, 14, 14, 64), dtype=float32)
        d2 = d2 + d_b2 # Tensor("add_5:0", shape=(?, 14, 14, 64), dtype=float32) .....Tensor("add_9:0", shape=(50, 14, 14, 64), dtype=float32)
        d2 = tf.nn.relu(d2) # Tensor("Relu_4:0", shape=(?, 14, 14, 64), dtype=float32) .......Tensor("Relu_7:0", shape=(50, 14, 14, 64), dtype=float32)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Tensor("AvgPool_1:0", shape=(?, 7, 7, 64), dtype=float32) .....Tensor("AvgPool_3:0", shape=(50, 7, 7, 64), dtype=float32)

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02)) # Tensor("d_w3/read:0", shape=(3136, 1024), dtype=float32) ......Tensor("d_w3/read:0", shape=(3136, 1024), dtype=float32)
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0)) # Tensor("d_b3/read:0", shape=(1024,), dtype=float32) .......Tensor("d_b3/read:0", shape=(1024,), dtype=float32)
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64]) # Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32) ......Tensor("Reshape_2:0", shape=(50, 3136), dtype=float32)
        d3 = tf.matmul(d3, d_w3) # Tensor("MatMul_1:0", shape=(?, 1024), dtype=float32) .......Tensor("MatMul_3:0", shape=(50, 1024), dtype=float32)
        d3 = d3 + d_b3 # Tensor("add_6:0", shape=(?, 1024), dtype=float32) .....Tensor("add_10:0", shape=(50, 1024), dtype=float32)
        d3 = tf.nn.relu(d3) # Tensor("Relu_5:0", shape=(?, 1024), dtype=float32) ......Tensor("Relu_8:0", shape=(50, 1024), dtype=float32)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02)) # Tensor("d_w4/read:0", shape=(1024, 1), dtype=float32) .......Tensor("d_w4/read:0", shape=(1024, 1), dtype=float32)
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0)) # Tensor("d_b4/read:0", shape=(1,), dtype=float32) .....Tensor("d_b4/read:0", shape=(1,), dtype=float32)
        d4 = tf.matmul(d3, d_w4) + d_b4 # Tensor("add_7:0", shape=(?, 1), dtype=float32) ......Tensor("add_11:0", shape=(50, 1), dtype=float32)

        # d4 contains unscaled values
        return d4

# Define the generator network fast
# 一般卷积中[5,5,1,32] 前两个纬度是patch的大小，第3个是输入通道数目，第4个是输出通道数目
# 输入一个N为的向量，转换成56*56,最后生成28*28的图像
def generator(batch_size, z_dim):
    # 生成截断正态分布随机数
    z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1 # 矩阵乘法
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

z_dimensions = 100
batch_size = 50

# 作为鉴别器的输入
x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

# 保持生成器生成的图片
Gz = generator(batch_size, z_dimensions)
# Gz holds the generated images

# Dx 为鉴别器预测MNIST图片为真的概率
Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

# Dg 为鉴别器预测生成图片的概率
Dg = discriminator(Gz, reuse_variables=True)  # 这里调用函数
# Dg will hold discriminator prediction probabilities for generated images


# 鉴别器的目的正确的把MNIST图片标记为真，生成图片标记为假
# 为鉴别器计算两个losses，对真实图片的与1比较，对生成图片与0比较
# 鉴别器后面没有softmax和sigmoid层
# sigmoid_cross_entropy_with_logits operates on unscaled values rather than probability values from 0 to 1
# GAN在鉴别器饱和 或者过于自信（对生成器的输出预测值为0）时失败，这导致鉴别器没有有效的梯度下降

# reduce_mean 取由交叉熵函数反回矩阵中所有分量的平均值
# Define losses

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

# 生成器的loss函数，生成器希望当其输出图片时鉴别器的输出接近1
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))


# 生成器的优化函数需要更新生成器的权重参数，而不要生成鉴别器的参数
# 训练鉴别器时同样要保持生成器参数
# Define variable lists
# 把所有定义为trainable=True的变量以list返回
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Define the optimizers
# Train the discriminator
# Adam 是GAN长用的优化算法，自适的学习速率和动量
# 为鉴别器设置两个优化函数，一个鉴别真实图片，一个鉴别生图片
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()
# 当tf.get_variable_scope().reuse == False时，作用域就是为创建新变量所设置的
# 当tf.get_variable_scope().reuse == True时，作用域是为重用变量所设置
# test_vars = tf.get_variable_scope().reuse_variables()
# print(test_vars)
# 调用tf.get_variable(name),得到一个已经存在名字为name的变量

sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# 为了训练鉴别器，从MNIST取一批图片作为正样本，用生成图片作为负样本
# 生成器的改进输出，鉴别器同样改进以将改进输出后的生成器所生成的图片标记为假
# Pre-train discriminator
# for i in range(300):
#     real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
#     _, __ = sess.run([d_trainer_real, d_trainer_fake],
#                                            {x_placeholder: real_image_batch})

# Train generator and discriminator together


for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

    # print('0',sess.run(d_vars[0]))
    # print('1',sess.run(d_vars[1]))
    # Train discriminator on both real and fake images
    print('d_trainer_real,d_trainer_fake')
    _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: real_image_batch})
    print('g_trainer')
    # Train generator
    _ = sess.run(g_trainer)

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        summary = sess.run(merged, {x_placeholder: real_image_batch})
        writer.add_summary(summary, i)


# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, 'pretrained-model/pretrained_gan.ckpt')
#     z_batch = np.random.normal(0, 1, size=[10, z_dimensions])
#     # z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
#     z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
#     generated_images = generator(10, z_dimensions)
#     images = sess.run(generated_images, {z_placeholder: z_batch})
#     for i in range(10):
#         plt.imshow(images[i].reshape([28, 28]), cmap='Greys')
#         plt.show()