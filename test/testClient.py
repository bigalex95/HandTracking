import imagiz
import queue
import os
import threading
import time
import cv2
import numpy as np
import socketio
import base64
import tensorflow as tf
import multiprocessing

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
exitFlag = 0
queueLock = threading.Lock()
inputPix2PixQueue1 = queue.Queue(5)
inputPix2PixQueue2 = queue.Queue(5)
inputFrameQueue = multiprocessing.Queue(5)
pix2pixQueue1 = queue.Queue(5)
pix2pixQueue2 = queue.Queue(5)
resizedTFQueue = multiprocessing.Queue(5)
handQueue = queue.Queue(5)
# client1 = imagiz.TCP_Client(
#     server_ip='localhost', server_port=5550, client_name='cc1')
# client2 = imagiz.TCP_Client(
#     server_ip='localhost', server_port=5550, client_name='cc2')
# client3 = imagiz.TCP_Client(
#     server_ip='localhost', server_port=5550, client_name='cc3')
sio = socketio.Client()
sio.connect('http://10.42.0.1:3000', namespaces=['/'])
SIZE = 256
NORM = 127.5
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


@sio.event
def connect():
    print("I'm connected!")


@sio.event
def connect_error():
    print("The connection failed!")


@sio.event
def disconnect():
    print("I'm disconnected!")

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
class myThread(threading.Thread):
    def __init__(self, name, function, iq, oq=None):
        threading.Thread.__init__(self)
        self.name = name
        self.function = function
        self.iq = iq
        self.oq = oq

    def run(self):
        print(style.YELLOW + "Starting " + self.name)
        if self.oq:
            self.function(self.iq, self.oq)
        else:
            self.function(self.iq)
        print(style.GREEN + "Exiting " + self.name)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
class myProcess(multiprocessing.Process):
    def __init__(self, name, function, iq, oq=None):
        multiprocessing.Process.__init__(self)
        self.name = name
        self.function = function
        self.iq = iq
        self.oq = oq

    def run(self):
        print(style.YELLOW + "Starting " + self.name)
        if self.oq:
            self.function(self.iq, self.oq)
        else:
            self.function(self.iq)
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


def resize(iq, oq):
    while not exitFlag:
        if not iq.empty():
            image = iq.get()
            # print(image.shape)
            # print(type(image))
            imgTF = tf.convert_to_tensor(image)
            imgTF = tf.image.resize(imgTF, (256, 256))
            imgTF = tf.cast(imgTF,  dtype=tf.uint8)
            if not oq.full():
                oq.put(imgTF.numpy())
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def send_to_server(iq, cl):
    while not exitFlag:
        if not iq.empty():
            # from image to binary buffer
            _, frame = cv2.imencode('.jpg', iq.get())
            # convert to base64 format
            data = base64.b64encode(frame)
            # send to server
            sio.emit(cl, data, namespace='/')

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    global exitFlag
    threads = []
    cameras = []
    cap1 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=0), device=cv2.CAP_GSTREAMER).start()
    # cap2 = WebcamVideoStream(src=gstreamer_pipeline(
    #     sensor_id=1), device=cv2.CAP_GSTREAMER).start()
    cameras.append(cap1)
    # cameras.append(cap2)
    # Defining and start Threads
    clientTH1 = myThread("client 1", send_to_server, resizedTFQueue, 'data1')
    # clientTH2 = myThread("client 2", send_to_server, pix2pixQueue2, 'data2')
    # clientTH3 = myThread("client 3", send_to_server, handQueue, 'data3')
    reizeTH = myThread("Resize Thread", resize,
                       inputFrameQueue, resizedTFQueue)
    # pix2pixTH1 = myThread(
    #     "pix2pix1 Thread", get_from_model, inputPix2PixQueue1, pix2pixQueue1)
    # pix2pixTH2 = myThread(
    #     "pix2pix2 Thread", get_from_model, inputPix2PixQueue2, pix2pixQueue2)
    # handTH = myThread("hand Pose Thread", execute, resizedTFQueue, handQueue)

    threads.append(clientTH1)
    # threads.append(clientTH2)
    # threads.append(clientTH3)
    # threads.append(handTH)
    # threads.append(pix2pixTH2)
    # threads.append(pix2pixTH1)
    threads.append(reizeTH)

    for t in threads:
        t.start()

    try:
        while True:
            frame1 = cap1.read()
            # frame2 = cap2.read()
            # if frame1:
            if not inputFrameQueue.full():
                inputFrameQueue.put(frame1)
    except Exception as e:
        print(style.RED + str(e))
        for c in cameras:
            c.stop()
        # Notify threads it's time to exit
        exitFlag = 1
        # Wait for all threads to complete
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        for c in cameras:
            c.stop()
        # Notify threads it's time to exit
        exitFlag = 1
        # Wait for all threads to complete
        for t in threads:
            t.join()

    # Notify threads it's time to exit
    exitFlag = 1
    # Wait for all threads to complete
    for t in threads:
        t.join()
    for c in cameras:
        c.stop()
    print(style.GREEN + "Exiting Main Thread")


if __name__ == "__main__":
    main()
