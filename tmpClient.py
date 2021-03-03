import threading
import ctypes
import time
import cv2
import tensorflow as tf
import numpy as np

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import os
from preprocessdata import preprocessdata
import queue
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
inputPix2PixQueue = queue.Queue(10)
inputFrameQueue = queue.Queue(10)
pix2pixQueue = queue.Queue(10)
resizedTFQueue = queue.Queue(10)
handQueue = queue.Queue(10)
# pix2pixFrame = np.zeros((256, 256, 1))
# gray = np.zeros((480, 640, 1))
SIZE = 256
NORM = 127.5
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


class myThread(threading.Thread):
    def __init__(self, name, function):
        threading.Thread.__init__(self)
        self.name = name
        self.function = function

    def run(self):
        print(style.YELLOW + "Starting " + self.name)
        self.function(self.name)
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


def draw_joints(image, joints):
    count = 0
    for i in joints:
        if i == [0, 0]:
            count += 1
    if count >= 19:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, (255, 0, 255), 1)
    for i in hand_pose['skeleton']:
        if joints[i[0]-1][0] == 0 or joints[i[1]-1][0] == 0:
            break
        cv2.line(image, (joints[i[0]-1][0], joints[i[0]-1][1]),
                 (joints[i[1]-1][0], joints[i[1]-1][1]), (0, 255, 0), 1)


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


def execute(threadName):
    global device
    while not exitFlag:
        if not resizedTFQueue.empty():
            image = resizedTFQueue.get()
            print(image.shape)
            print(type(image))
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
            # draw_joints(image, joints)
            # cv2.imshow("execute", image)
            if not handQueue.full():
                handQueue.put(joints)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def resize(threadName):
    while not exitFlag:
        if not inputFrameQueue.empty():
            image = inputFrameQueue.get()
            print(image.shape)
            print(type(image))
            imgTF = tf.convert_to_tensor(image)
            imgTF = tf.image.resize(imgTF, (WIDTH, HEIGHT))
            imgTF = tf.cast(imgTF,  dtype=tf.uint8)
            if not resizedTFQueue.full():
                resizedTFQueue.put(imgTF.numpy())

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


def get_from_model(threadName):
    while not exitFlag:
        if not inputPix2PixQueue.empty():
            image = inputPix2PixQueue.get()
            print(image.shape)
            print(type(image))
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
            if not pix2pixQueue.full():
                pix2pixQueue.put(np.array(pil_image))


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    cap = WebcamVideoStream(src=gstreamer_pipeline(
        sensor_id=0), device=cv2.CAP_GSTREAMER).start()
    # Defining and start Threads
    threads = []
    global exitFlag

    reizeTH = myThread("Resize Thread", resize)
    reizeTH.start()
    threads.append(reizeTH)
    pix2pixTH = myThread("pix2pix Thread", get_from_model)
    pix2pixTH.start()
    threads.append(pix2pixTH)
    handTH = myThread("hand Pose Thread", execute)
    handTH.start()
    threads.append(handTH)

    try:
        while True:
            t0 = time.time()
            frame = cap.read()
            inputFrameQueue.put(frame)
            inputPix2PixQueue.put(frame)
            if not pix2pixQueue.empty():
                pix2pixQueue.get()
            if not handQueue.empty():
                handQueue.get()
            print(style.WHITE + "pix2pixQueue = " +
                  str(pix2pixQueue.qsize()))
            print(style.WHITE + "handQueue = " + str(handQueue.qsize()))
            print(style.WHITE + "resizedTFQueue = " +
                  str(resizedTFQueue.qsize()))
            print(style.WHITE + "inputFrameQueue = " +
                  str(inputFrameQueue.qsize()))
            print(style.WHITE + "inputPix2PixQueue = " +
                  str(inputPix2PixQueue.qsize()))
            t1 = time.time()
            print(style.BLUE + str(1 / (t1 - t0)))
            if cv2.waitKey(1) == 27:
                break
    except Exception as e:
        print(style.RED + str(e))
        cv2.destroyAllWindows()
        cap.stop()
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
    cap.stop()
    cv2.destroyAllWindows()
    print(style.GREEN + "Exiting Main Thread")


if __name__ == "__main__":
    main()
