from imutils.video import FPS
from threading import Thread
import numpy as np
import cv2
import imagiz
import tensorflow as tf
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(
            logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class WebcamVideoStream:
    def __init__(self, src=0, device=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src, device)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3840,
    capture_height=2160,
    display_width=1920,  # 2560
    display_height=1080,  # 1440
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def crop(img, offset_height, offset_width, target_height, target_width):
    """
    Args
    image 	4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    offset_height 	Vertical coordinate of the top-left corner of the result in the input.
    offset_width 	Horizontal coordinate of the top-left corner of the result in the input.
    target_height 	Height of the result.
    target_width 	Width of the result. 

    Returns
    If image was 4-D, a 4-D float Tensor of shape [batch, target_height, target_width, channels] 
    If image was 3-D, a 3-D float Tensor of shape [target_height, target_width, channels]
    """
    return tf.image.crop_to_bounding_box(
        img, offset_height, offset_width, target_height, target_width)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def resize(img, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
           antialias=False, name=None):
    """
    Args
    images 	4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    size 	A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
    method 	An image.ResizeMethod, or string equivalent. Defaults to bilinear.

    Returns
    If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels]. 
    If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels]. 
    """
    return tf.image.resize(img, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
                           antialias=False, name=None)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def pix2pix(img, size, norm, model):
    input_image = tf.cast(img, tf.float32)
    input_image = tf.image.resize(input_image, [size, size],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / norm) - 1
    ext_image = tf.expand_dims(input_image, axis=0)
    prediction = model(ext_image, training=True)
    pil_image = tf.keras.preprocessing.image.array_to_img(
        prediction[0])
    return pil_image
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    parser = argparse.ArgumentParser(
        description='pix2pix checkpoint to SavedModel.')
    parser.add_argument('--size', dest='size',
                        help='size of model', type=int, default=256)
    parser.add_argument('--cropSize', dest='cropSize',
                        help='', type=int, default=1080)
    parser.add_argument('--sensor', dest='sensor',
                        help='', type=int, default=0)
    parser.add_argument('--clientName', dest='clientName',
                        help='', type=str, default='cc0')
    parser.add_argument('--serverIP', dest='serverIP',
                        help='', type=str, default='10.42.0.1')
    parser.add_argument('--serverPORT', dest='serverPORT',
                        help='', type=int, default=5550)
    parser.add_argument('--model', dest='model',
                        help='', type=str, default='./model/pix2pixTF-TRT')
    parser.add_argument('--resize', dest='resize',
                        help='', type=str2bool, const=True, default=False)
    parser.add_argument('--crop', dest='crop',
                        help='', type=str2bool, const=True, default=False)
    parser.add_argument('--pix2pix', dest='pix2pix',
                        help='', type=str2bool, const=True, default=False)
    args = parser.parse_args()
    print(args)

    if args.pix2pix:
        generator = tf.saved_model.load("./model/pix2pixTF-TRT")
        norm = (args.size/2)-0.5

    vs = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=0), device=cv2.CAP_GSTREAMER).start()

    client = imagiz.TCP_Client(
        server_ip=args.serverIP, server_port=args.serverPORT, client_name=args.clientName)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    fps = FPS().start()
    while True:
        try:
            image = vs.read()
            image_tf = tf.convert_to_tensor(image)
            if args.crop:
                image_tf = crop(image_tf, 0, 0, args.cropSize, args.cropSize)
                image = image_tf.numpy()
            if args.resize:
                image_tf = resize(image_tf, args.size)
                image = image_tf.numpy()
            if args.pix2pix:
                image = pix2pix(image_tf, args.size, norm, generator)
            # _, image = cv2.imencode('.jpg', image, encode_param)
            response = client.send(image)
            # print(response)
            fps.update()
        except Exception as e:
            print(e)
            vs.stop()
            break
        except KeyboardInterrupt:
            print("Keyboard stopped")
            vs.stop()
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.stop()
    print("exit main")


if __name__ == "__main__":
    main()
