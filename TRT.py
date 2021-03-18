import tensorflow as tf
import cv2
import time

input_saved_model_dir = './resize_modelTF'
output_saved_model_dir = './resize_modelTF-TRT'

img = cv2.imread('test.jpg')
image = tf.convert_to_tensor(img,  dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

model_original = tf.saved_model.load(input_saved_model_dir)

start = time.time()
for i in range(10000):
    result = model_original(image)
print(time.time() - start)
cv2.imwrite('result2.jpg',  result[0].numpy())
