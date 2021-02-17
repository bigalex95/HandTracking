#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

# In[36]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Pix2Pix

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/generative/pix2pix"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This notebook demonstrates image to image translation using conditional GAN's, as described in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). Using this technique we can colorize black and white photos, convert google maps to google earth, etc. Here, we convert building facades to real buildings.
# 
# In example, we will use the [CMP Facade Database](http://cmp.felk.cvut.cz/~tylecr1/facade/), helpfully provided by the [Center for Machine Perception](http://cmp.felk.cvut.cz/) at the [Czech Technical University in Prague](https://www.cvut.cz/). To keep our example short, we will use a preprocessed [copy](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/) of this dataset, created by the authors of the [paper](https://arxiv.org/abs/1611.07004) above.
# 
# Each epoch takes around 15 seconds on a single V100 GPU.
# 
# Below is the output generated after training the model for 200 epochs.
# 
# ![sample output_1](https://www.tensorflow.org/images/gan/pix2pix_1.png)
# ![sample output_2](https://www.tensorflow.org/images/gan/pix2pix_2.png)

# ## Import TensorFlow and other libraries

# In[37]:


import tensorflow as tf

import os
import time

import cv2
import numpy as np
from PIL import Image


# In[39]:


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


# ## Build the Generator
#   * The architecture of generator is a modified U-Net.
#   * Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
#   * Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
#   * There are skip connections between the encoder and decoder (as in U-Net).
# 

# In[40]:


OUTPUT_CHANNELS = 3


# In[41]:


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# In[42]:


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# In[43]:


def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# In[44]:


generator = Generator()
tf.keras.utils.plot_model(generator, to_file='generatorModel.png', show_shapes=True, dpi=64)


# * **Generator loss**
#   * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
#   * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
#   * This allows the generated image to become structurally similar to the target image.
#   * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

# The training procedure for the generator is shown below:

# In[45]:


LAMBDA = 100


# In[46]:


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# ![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)
# 

# ## Build the Discriminator
#   * The Discriminator is a PatchGAN.
#   * Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
#   * The shape of the output after the last layer is (batch_size, 30, 30, 1)
#   * Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
#   * Discriminator receives 2 inputs.
#     * Input image and the target image, which it should classify as real.
#     * Input image and the generated image (output of generator), which it should classify as fake.
#     * We concatenate these 2 inputs together in the code (`tf.concat([inp, tar], axis=-1)`)

# In[47]:


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# In[48]:


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, to_file='discriminatorModel.png', show_shapes=True, dpi=64)


# **Discriminator loss**
#   * The discriminator loss function takes 2 inputs; **real images, generated images**
#   * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
#   * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
#   * Then the total_loss is the sum of real_loss and the generated_loss
# 

# In[49]:


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[50]:


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


# The training procedure for the discriminator is shown below.
# 
# To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

# ![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)
# 

# ## Define the Optimizers and Checkpoint-saver
# 

# In[51]:


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# In[52]:


checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Generate Images
# 
# Write a function to plot some images during training.
# 
# * We pass images from the test dataset to the generator.
# * The generator will then translate the input image into the output.
# * Last step is to plot the predictions and **voila!**

# Note: The `training=True` is intentional here since
# we want the batch statistics while running the model
# on the test dataset. If we use training=False, we will get
# the accumulated statistics learned from the training dataset
# (which we don't want)

# In[53]:


def generate_images(model, test_input):
  start = time.time()
  prediction = model(test_input, training=True)
  # print('Time taken for frame {} sec\n'.format(time.time()-start))

  return prediction[0]


# ## Restore the latest checkpoint and test

# In[54]:


checkpoint_dir = 'training_checkpoints'
# get_ipython().system('ls {checkpoint_dir}')


# In[55]:


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## Generate using Camera Input

# Interpreting the logs from a GAN is more subtle than a simple classification or regression model. Things to look for::
# 
# * Check that neither model has "won". If either the `gen_gan_loss` or the `disc_loss` gets very low it's an indicator that this model is dominating the other, and you are not successfully training the combined model.
# * The value `log(2) = 0.69` is a good reference point for these losses, as it indicates a perplexity of 2: That the discriminator is on average equally uncertain about the two options.
# * For the `disc_loss` a value below `0.69` means the discriminator is doing better than random, on the combined set of real+generated images.
# * For the `gen_gan_loss` a value below `0.69` means the generator i doing better than random at foolding the descriminator.
# * As training progresses the `gen_l1_loss` should go down.

# In[56]:


def load_from_video(image_file):

    input_image = tf.cast(image_file, tf.float32)
    
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    input_image = (input_image / 127.5) - 1
    
    return input_image


# In[58]:


cap = cv2.VideoCapture(0)
  
# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = 512 
frame_height = 512 
# frame_width = int(webcam.get(3)) 
# frame_height = int(webcam.get(4)) 

size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
# result = cv2.VideoWriter('fulllayer.avi',  
#                         cv2.VideoWriter_fourcc(*'XVID'), 
#                         15, size) 


while True:
    start = time.time()
    ret, frame = cap.read()
    resized = cv2.resize(frame, (256, 256))
    # print('point 1')
    input_image = load_from_video(frame)
    # print('point 2', input_image)
    ext_image = tf.expand_dims(input_image, axis=0)
    # print('point 3', input_image)
    generated_image = generate_images(generator, ext_image)
    pil_image = tf.keras.preprocessing.image.array_to_img(generated_image)
    imtemp = pil_image.copy()
    # print(imtemp)
    #review = np.array(imtemp)
    review = cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR)
    img_concate_Hori=np.concatenate((resized,review),axis=1)
    # cv2.imshow('Live1 Video', frame)
    # cv2.imshow('Live2 Video', review)
    cv2.imshow('Live Video', img_concate_Hori)
    print('Time taken for frame {} sec\n'.format(time.time()-start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


