import tensorflow as tf
import cv2
import PIL
import numpy as np
import time

generator = tf.saved_model.load("./model/saved_modelTF")


def generate_images(model, test_input):
    prediction = model(test_input, training=True)

    return prediction[0]


def load_from_video(image_file):
    input_image = tf.cast(image_file, tf.float32)
    input_image = tf.image.resize(input_image, [256, 256],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / 127.5) - 1

    return input_image


cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    ret, frame = cap.read()
    # resized = cv2.resize(frame, (256, 256))
    input_image = load_from_video(frame)
    ext_image = tf.expand_dims(input_image, axis=0)
    generated_image = generate_images(generator, ext_image)
    pil_image = tf.keras.preprocessing.image.array_to_img(generated_image)
    # imtemp = pil_image.copy()
    #review = np.array(imtemp)
    review = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    # img_concate_Hori = np.concatenate((resized, review), axis=1)
    # cv2.imshow('Live1 Video', frame)
    cv2.imshow('Live2 Video', review)
    cv2.imshow('Live1 Video', np.array(pil_image))
    # cv2.imshow('Live Video', img_concate_Hori)
    print('Time taken for frame {} sec\n'.format(time.time()-start))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
