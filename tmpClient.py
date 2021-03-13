# import trt_pose.coco
# from trt_pose.parse_objects import ParseObjects
# from trt_pose.draw_objects import DrawObjects
# import trt_pose.models
# from torch2trt import TRTModule
# import torch2trt
# import json
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
generator = tf.saved_model.load("./model/pix2pixTF-TRT")
# <==================================================================>
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
exitFlag = 0
queueLock = threading.Lock()
inputPix2PixQueue1 = queue.Queue(5)
inputPix2PixQueue2 = queue.Queue(5)
inputFrameQueue = queue.Queue(5)
pix2pixQueue1 = queue.Queue(5)
pix2pixQueue2 = queue.Queue(5)
resizedTFQueue = queue.Queue(5)
handQueue = queue.Queue(5)
client1 = imagiz.TCP_Client(
    server_ip='localhost', server_port=5550, client_name='cc1')
# client2 = imagiz.TCP_Client(
#     server_ip='localhost', server_port=5550, client_name='cc2')
# client3 = imagiz.TCP_Client(
#     server_ip='localhost', server_port=5550, client_name='cc3')
SIZE = 256
NORM = 127.5
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
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# with open('hand_pose.json', 'r') as f:
#     hand_pose = json.load(f)

# topology = trt_pose.coco.coco_category_to_topology(hand_pose)

# num_parts = len(hand_pose['keypoints'])
# num_links = len(hand_pose['skeleton'])

# model = trt_pose.models.resnet18_baseline_att(
#     num_parts, 2 * num_links).cuda().eval()

# WIDTH = 224
# HEIGHT = 224
# data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

# if not os.path.exists('./model/hand_pose_resnet18_att_244_244_trt.pth'):
#     MODEL_WEIGHTS = './model/hand_pose_resnet18_att_244_244.pth'
#     model.load_state_dict(torch.load(MODEL_WEIGHTS))
#     import torch2trt
#     model_trt = torch2trt.torch2trt(
#         model, [data], fp16_mode=True, max_workspace_size=1 << 25)
#     OPTIMIZED_MODEL = './model/hand_pose_resnet18_att_244_244_trt.pth'
#     torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

# OPTIMIZED_MODEL = './model/hand_pose_resnet18_att_244_244_trt.pth'

# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# parse_objects = ParseObjects(
#     topology, cmap_threshold=0.15, link_threshold=0.15)
# draw_objects = DrawObjects(topology)

# mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
# std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
# device = torch.device('cuda')

# preprocessdata = preprocessdata(topology, num_parts)
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
    # print(style.RED + str(oq.maxsize))
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
                print(arrjoints)
                oq.put(arrjoints)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def resize(iq, oq):
    # print(style.RED + str(oq.maxsize))
    while not exitFlag:
        if not iq.empty():
            image = iq.get()
            # print(image.shape)
            # print(type(image))
            imgTF = tf.convert_to_tensor(image)
            imgTF = tf.image.resize(imgTF, (WIDTH, HEIGHT))
            imgTF = tf.cast(imgTF,  dtype=tf.uint8)
            if not oq.full():
                oq.put(imgTF.numpy())

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def generate_images(model, image):
    prediction = model(image, training=True)
    return prediction[0]


def load_from_video(image):
    input_image = tf.cast(image, tf.float32)
    input_image = tf.image.resize(input_image, [SIZE, SIZE],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / NORM) - 1
    return input_image


def get_from_model(iq, oq):
    # print(style.RED + str(oq.maxsize))
    while not exitFlag:
        if not iq.empty():
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

def send_to_server(iq, cl):
    print(iq.maxsize)
    while not exitFlag:
        if not iq.empty():
            message = iq.get()
            # print(message.image)
            cl.send(message)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    global exitFlag
    threads = []
    cameras = []
    cap1 = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=2), device=cv2.CAP_GSTREAMER).start()
    # cap2 = WebcamVideoStream(src=gstreamer_pipeline(
    #     sensor_id=1), device=cv2.CAP_GSTREAMER).start()
    cameras.append(cap1)
    # cameras.append(cap2)
    # Defining and start Threads
    clientTH1 = myThread("client 1", send_to_server, pix2pixQueue1, client1)
    # clientTH2 = myThread("client 2", send_to_server, pix2pixQueue2, client2)
    # clientTH3 = myThread("client 3", send_to_server, handQueue, client3)
    # reizeTH = myThread("Resize Thread", resize,
    #                    inputFrameQueue, resizedTFQueue)
    pix2pixTH1 = myThread(
        "pix2pix1 Thread", get_from_model, inputPix2PixQueue1, pix2pixQueue1)
    # pix2pixTH2 = myThread(
    #     "pix2pix2 Thread", get_from_model, inputPix2PixQueue2, pix2pixQueue2)
    # handTH = myThread("hand Pose Thread", execute, resizedTFQueue, handQueue)

    threads.append(clientTH1)
    # threads.append(clientTH2)
    # threads.append(clientTH3)
    # threads.append(handTH)
    # threads.append(pix2pixTH2)
    threads.append(pix2pixTH1)
    # threads.append(reizeTH)

    for t in threads:
        t.start()

    try:
        while True:
            frame1 = cap1.read()
            # frame2 = cap2.read()
            # if not inputFrameQueue.full():
            #     inputFrameQueue.put(frame1)

            if inputPix2PixQueue1.full():
                tmp = inputPix2PixQueue1.get()
                inputPix2PixQueue1.put(frame1)

            # if not inputPix2PixQueue2.full():
            #     inputPix2PixQueue2.put(frame2)
            # print(style.YELLOW + "pix2pixQueue1 = " +
            #       str(pix2pixQueue1.qsize()))
            # print(style.YELLOW + "pix2pixQueue2 = " +
            #       str(pix2pixQueue2.qsize()))
            # print(style.YELLOW + "handQueue = " + str(handQueue.qsize()))
            # print(style.YELLOW + "resizedTFQueue = " +
            #       str(resizedTFQueue.qsize()))
            # print(style.YELLOW + "inputFrameQueue = " +
            #       str(inputFrameQueue.qsize()))
            # print(style.YELLOW + "inputPix2PixQueue1 = " +
            #       str(inputPix2PixQueue1.qsize()))
            # print(style.YELLOW + "inputPix2PixQueue2 = " +
            #       str(inputPix2PixQueue2.qsize()))
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
