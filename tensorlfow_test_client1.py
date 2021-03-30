import numpy as np
import cv2
import time
import threading
from multiprocessing import Process, Queue
import os
import queue
import imagiz
# import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# System call
os.system("")

# Class of different styles


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Pix2Pix variables declaration
# <==================================================================>
# generator = tf.saved_model.load("./model/pix2pixTF-TRT512")
# <==================================================================>
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
exitFlag = 0
queueLock = threading.Lock()
inputPix2PixQueue1 = Queue(3)
outputPix2PixQueue1 = Queue(3)
inputPix2PixQueue2 = Queue(3)
outputPix2PixQueue2 = Queue(3)
client1 = imagiz.TCP_Client(
    server_ip='10.42.0.1', server_port=5550, client_name='cc1')
client2 = imagiz.TCP_Client(
    server_ip='10.42.0.1', server_port=5550, client_name='cc2')
SIZE = 512
NORM = 255.5
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


class myProcess(Process):
    def __init__(self, name, function, iq, oq=None):
        Process.__init__(self)
        self.name = name
        self.function = function
        self.iq = iq
        self.oq = oq

    def run(self):
        print(style.YELLOW + "Starting " + self.name)
        if self.oq:
            self.function(self.name, self.iq, self.oq)
        else:
            self.function(self.name, self.iq)
        print(style.GREEN + "Exiting " + self.name)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


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
        threading.Thread(target=self.update, args=()).start()
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


# def crop(img, offset_height, offset_width, target_height, target_width):
#     """
#     Args
#     image 	4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
#     offset_height 	Vertical coordinate of the top-left corner of the result in the input.
#     offset_width 	Horizontal coordinate of the top-left corner of the result in the input.
#     target_height 	Height of the result.
#     target_width 	Width of the result.

#     Returns
#     If image was 4-D, a 4-D float Tensor of shape [batch, target_height, target_width, channels]
#     If image was 3-D, a 3-D float Tensor of shape [target_height, target_width, channels]
#     """
#     return tf.image.crop_to_bounding_box(
#         img, offset_height, offset_width, target_height, target_width)
# # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


# def resize(img, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
#            antialias=False, name=None):
#     """
#     Args
#     images 	4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
#     size 	A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
#     method 	An image.ResizeMethod, or string equivalent. Defaults to bilinear.

#     Returns
#     If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
#     If images was 3-D, a 3-D float Tensor of shape [new_height, new_width, channels].
#     """
#     return tf.image.resize(img, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
#                            antialias=False, name=None)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def get_from_model(name, iq, oq):
    print(style.BLUE + name)
    import tensorflow as tf

    generator = tf.saved_model.load("./model/pix2pixTF-TRT512")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices(
                'GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(style.RED + e)
    print(style.CYAN + name)
    while not exitFlag:
        if not iq.empty():
            print(style.GREEN + name)
            image = iq.get()
            # print(image.shape)
            # print(type(image))
            input_image = tf.cast(image, tf.float32)
            input_image = tf.image.resize(input_image, [SIZE, SIZE],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            input_image = (input_image / NORM) - 1
            # input_image = load_from_video(image)
            ext_image = tf.expand_dims(input_image, axis=0)
            prediction = generator(ext_image, training=True)
            # generated_image = generate_images(generator, ext_image)
            # pil_image = tf.keras.preprocessing.image.array_to_img(
            #     generated_image)
            pil_image = tf.keras.preprocessing.image.array_to_img(
                prediction[0])
            if not oq.full():
                oq.put(np.array(pil_image))
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def send_to_imagiz_server(name, iq, cl):
    print(style.BLUE + name)
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while not exitFlag:
        if not iq.empty():
            print(style.GREEN + name)
            # from image to binary buffer
            image = iq.get()
            # _, image = cv2.imencode('.jpg', iq.get(), encode_param)
            res = cl.send(image)
            # print(res)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    global exitFlag
    process = []
    cameras = []
    cap1 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=0), device=cv2.CAP_GSTREAMER).start()
    cap2 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=1), device=cv2.CAP_GSTREAMER).start()
    cameras.append(cap1)
    cameras.append(cap2)
    clientTH1 = myProcess("client 1", send_to_imagiz_server,
                          outputPix2PixQueue1, client1)
    clientTH2 = myProcess("client 2", send_to_imagiz_server,
                          outputPix2PixQueue2, client2)

    pix2pixTH1 = myProcess(
        "pix2pix1 Thread", get_from_model, inputPix2PixQueue1, outputPix2PixQueue1)
    pix2pixTH2 = myProcess(
        "pix2pix2 Thread", get_from_model, inputPix2PixQueue2, outputPix2PixQueue2)

    process.append(clientTH1)
    process.append(pix2pixTH1)
    process.append(clientTH2)
    process.append(pix2pixTH2)

    for t in process:
        t.start()

    try:
        while True:
            frame1 = cap1.read()
            frame2 = cap2.read()

            if not inputPix2PixQueue1.full():
                inputPix2PixQueue1.put(frame1)
            if not inputPix2PixQueue2.full():
                inputPix2PixQueue2.put(frame2)

    except Exception as e:
        print(style.RED + str(e))
        for c in cameras:
            c.stop()
        # Notify threads it's time to exit
        exitFlag = 1
        # Wait for all threads to complete
        for t in process:
            t.join()
    except KeyboardInterrupt:
        for c in cameras:
            c.stop()
        # Notify threads it's time to exit
        exitFlag = 1
        # Wait for all threads to complete
        for t in process:
            t.join()

    # Notify threads it's time to exit
    exitFlag = 1
    # Wait for all threads to complete
    for t in process:
        t.join()
    for c in cameras:
        c.stop()
    print(style.GREEN + "Exiting Main Thread")


if __name__ == "__main__":
    main()
