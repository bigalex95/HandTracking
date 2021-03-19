import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
import trt_pose.models
from torch2trt import TRTModule
import torch2trt
import json
import imagiz
import queue
from preprocessdata import preprocessdata
import os
import PIL.Image
import torchvision.transforms as transforms
import torch
import threading
import ctypes
import time
import cv2
import tensorflow as tf
import numpy as np
import socketio
import base64

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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# global variables
# <==================================================================>
exitFlag = 0
queueLock = threading.Lock()
input1Queue = queue.Queue(3)
input2Queue = queue.Queue(3)
input3Queue = queue.Queue(3)
resize1Queue = queue.Queue(3)
resize2Queue = queue.Queue(3)
resize3Queue = queue.Queue(3)
jointsQueue = queue.Queue(3)
x0 = 0
y0 = 0
WIDTH = 1080
HEIGHT = 1080

# imagiz Config
# <==================================================================>
client1 = imagiz.TCP_Client(
    server_ip='10.42.0.1', server_port=5550, client_name='cc1')
client2 = imagiz.TCP_Client(
    server_ip='10.42.0.1', server_port=5550, client_name='cc2')
client3 = imagiz.TCP_Client(
    server_ip='10.42.0.1', server_port=5550, client_name='cc3')

# Socketio Config
# <==================================================================>
# sio = socketio.Client()
# sio.connect('http://10.42.0.1:3000', namespaces=['/'])


# @sio.event
# def connect():
#     print("I'm connected!")


# @sio.event
# def connect_error():
#     print("The connection failed!")


# @sio.event
# def disconnect():
#     print("I'm disconnected!")


# custom Thread class
# <==================================================================>


class startThread(threading.Thread):
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

# Camera Config
# <==================================================================>


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


# TRT Pose Detection variables declaration
# <==================================================================>
with open('hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)

num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()

WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if not os.path.exists('./model/hand_pose_resnet18_att_244_244_trt.pth'):
    MODEL_WEIGHTS = './model/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(
        model, [data], fp16_mode=True, max_workspace_size=1 << 25)
    OPTIMIZED_MODEL = './model/hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

OPTIMIZED_MODEL = './model/hand_pose_resnet18_att_244_244_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

parse_objects = ParseObjects(
    topology, cmap_threshold=0.15, link_threshold=0.15)
draw_objects = DrawObjects(topology)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

preprocessdata = preprocessdata(topology, num_parts)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = image[..., ::-1]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = PIL.Image.fromarray((image * 255).astype(np.uint8))
    image = PIL.Image.fromarray((image).astype(np.uint8))
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def execute(iq, oq):
    global device
    while not exitFlag:
        if not iq.empty():
            image = iq.get()
            # print(image.shape)
            # print(type(image))
            device = torch.device('cuda')
            data = image[..., ::-1]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = PIL.Image.fromarray((image * 255).astype(np.uint8))
            data = PIL.Image.fromarray((data).astype(np.uint8))
            data = transforms.functional.to_tensor(data).to(device)
            data.sub_(mean[:, None, None]).div_(std[:, None, None])
            # data = preprocess(image)
            cmap, paf = model_trt(data[None, ...])
            # cmap, paf = model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)
            joints = preprocessdata.joints_inference(
                image, counts, objects, peaks)
            if not oq.full():
                arrjoints = np.array(joints)
                oq.put(arrjoints)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def resize(iq, oq, size):
    while not exitFlag:
        if not iq.empty():
            image = iq.get()
            image = image[0:1080, 420:1080]
            image_tf = tf.convert_to_tensor(image)
            # image_tf = tf.image.crop_to_bounding_box(
            #     image_tf, y0, x0, HEIGHT, WIDTH)
            image_tf = tf.image.resize(image_tf, (size, size))
            image_tf = tf.cast(image_tf,  dtype=tf.uint8)
            if not oq.full():
                oq.put(image_tf.numpy())

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
def resize_renew(iq, oq, size):
    while not exitFlag:
        if not iq.empty():
            image = iq.get()
            iq.put(image)
            image = image[0:1080, 420:1080]
            image_tf = tf.convert_to_tensor(image)
            # image_tf = tf.image.crop_to_bounding_box(
            #     image_tf, y0, x0, HEIGHT, WIDTH)
            image_tf = tf.image.resize(image_tf, (size, size))
            image_tf = tf.cast(image_tf,  dtype=tf.uint8)
            if not oq.full():
                oq.put(image_tf.numpy())

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def send_to_imagiz_server(iq, cl):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while not exitFlag:
        if not iq.empty():
            # from image to binary buffer
            image = iq.get()
            # _, image = cv2.imencode('.jpg', iq.get(), encode_param)
            res = cl.send(image)
            # print(res)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def send_to_socketio_server(iq, cl):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while not exitFlag:
        if not iq.empty():
            # from image to binary buffer
            _, image = cv2.imencode('.jpg', iq.get(), encode_param)
            # convert to base64 format
            data = base64.b64encode(image)
            # send to server
            sio.emit(cl, data, namespace='/')

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    global exitFlag
    threads = []
    cameras = []
    # Defining and start Cameras
    cap1 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=0), device=cv2.CAP_GSTREAMER).start()
    cap2 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=1), device=cv2.CAP_GSTREAMER).start()
    cameras.append(cap1)
    cameras.append(cap2)
    # Defining and start Threads
    clientTH1 = startThread(
        "client 1", send_to_imagiz_server, resize1Queue, client1)
    clientTH2 = startThread(
        "client 2", send_to_imagiz_server, resize2Queue, client2)
    clientTH3 = startThread(
        "client 3", send_to_imagiz_server, jointsQueue, client3)
    handTH = startThread("hand Pose Thread", execute, resize3Queue, jointsQueue)
    resizeTH1 = threading.Thread(
        target=resize_renew, args=(input1Queue, resize1Queue, 256,))
    resizeTH2 = threading.Thread(
        target=resize, args=(input2Queue, resize2Queue, 256,))
    resizeTH3 = threading.Thread(
        target=resize, args=(input1Queue, resize3Queue, 224,))

    threads.append(clientTH1)
    threads.append(clientTH2)
    threads.append(clientTH3)
    threads.append(handTH)
    threads.append(resizeTH1)
    threads.append(resizeTH2)
    threads.append(resizeTH3)

    for t in threads:
        t.start()

    try:
        while True:
            frame1 = cap1.read()
            frame2 = cap2.read()
            if not input1Queue.full():
                input1Queue.put(frame1)

            if not input2Queue.full():
                input2Queue.put(frame2)

            # if not input3Queue.full():
            #     input3Queue.put(frame1)

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
