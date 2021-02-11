import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2

img = cv2.imread('test.jpg')
image = tf.convert_to_tensor(img)
print(image.shape)
print(image.dtype)

IMG_SIZE = 256

resize_model = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
])
resize_model.compile(optimizer="adam", loss="mean_squared_error")
test_input = np.random.random((1, 1257, 1245, 3))
print(test_input.shape)
history = resize_model.fit(test_input)

resize_model.summary()

result = resize_model(image)
cv2.imwrite('result.jpg',  result.numpy())
print(result.shape)
print(result.dtype)

# resize_model.save('resize_modelKeras')
tf.saved_model.save(resize_model,  './resize_modelTF')
