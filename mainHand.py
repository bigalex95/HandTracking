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
# from imutils.video import FPS

# TRT Pose Detection variables declaration
# <==================================================================>
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()

MODEL_WEIGHTS = './model/resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(
    model, [data], fp16_mode=True, max_workspace_size=1 << 25)

OPTIMIZED_MODEL = './model/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
# <==================================================================>


# Pix2Pix variables declaration
# <==================================================================>
generator = tf.saved_model.load("./model/pix2pixTF")
# <==================================================================>


class pix2pixThreading(threading.Thread):
    def __init__(self, size=256):
        threading.Thread.__init__(self)
        self.size = size
        self.norm = 127.5
        if self.size == 512:
            self.norm = 255.5

    def generate_images(self, model, image):
        prediction = model(image, training=True)
        return prediction[0]

    def load_from_video(self, image):
        input_image = tf.cast(image, tf.float32)
        input_image = tf.image.resize(input_image, [self.size, self.size],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_image = (input_image / self.norm) - 1
        return input_image

    def getFromModel(self, image):
        input_image = self.load_from_video(image)
        ext_image = tf.expand_dims(input_image, axis=0)
        generated_image = self.generate_images(generator, ext_image)
        pil_image = tf.keras.preprocessing.image.array_to_img(generated_image)
        return np.array(pil_image)


class poseThreading(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

    def preprocess(self, image):
        global device
        device = torch.device('cuda')
        image = image[..., ::-1]
        # image = PIL.Image.fromarray((image * 255).astype(np.uint8))
        image = PIL.Image.fromarray((image).astype(np.uint8))
        # image = PIL.Image.fromarray((image).astype(np.uint8))
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def execute(self, change, name="execute"):
        image = change['new']
        data = self.preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        # , cmap_threshold=0.15, link_threshold=0.15)
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(image, counts, objects, peaks)
        cv2.imshow(name, image)


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

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

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


class resizeThreading(threading.Thread):
    def __init__(self, size=256):
        threading.Thread.__init__(self)
        self.size = size
        self.img = None
        self.imgTF = None

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

    def set(self, img):
        print("SET")
        self.img = img

    def getTF(self):
        print("GET")
        return self.imgTF

    def get(self):
        print("GET")
        return self.img

    def getResizeTF(self, size=None):
        if size:
            self.size = size
        self.imgTF = tf.convert_to_tensor(self.img)
        self.imgTF = tf.image.resize(self.imgTF, (self.size, self.size))
        self.imgTF = tf.cast(self.imgTF,  dtype=tf.uint8)
        return self.imgTF

    def getResize(self, size=None):
        if size:
            self.size = size
        self.img = self.img[..., ::-1]
        self.img = cv2.resize(self.img, (self.size, self.size))
        return self.img


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


#cv2.namedWindow('execute', cv2.WINDOW_NORMAL)
#cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)


def main():
    try:
        vs1 = WebcamVideoStream(src=gstreamer_pipeline(
            sensor_id=0), device=cv2.CAP_GSTREAMER).start()
        poseT1 = poseThreading()
        poseT1.start()
        resizeT1 = resizeThreading(224)
        resizeT1.start()
        pix2pixT1 = pix2pixThreading()
        pix2pixT1.start()

        vs2 = WebcamVideoStream(src=gstreamer_pipeline(
            sensor_id=1), device=cv2.CAP_GSTREAMER).start()
        pix2pixT2 = pix2pixThreading()
        pix2pixT2.start()
        # cap = cv2.VideoCapture(4)
        print('start capturing')
        while True:
            t0 = time.time()
            frame1 = vs1.read()
            frame2 = vs2.read()
            # _, frame1 = cap.read()

            pix2pixImg1 = pix2pixT1.getFromModel(frame1)
            pix2pixImg2 = pix2pixT2.getFromModel(frame2)
            cv2.imshow("frame1", pix2pixImg1)
            cv2.imshow("frame2", pix2pixImg2)

            resizeT1.set(frame1)
            resizeTF1 = resizeT1.getResizeTF()
            poseT1.execute({'new': resizeTF1.numpy()}, "Execute1")
            t1 = time.time()
            print(1 / (t1 - t0))
            if cv2.waitKey(1) == 27:
                break
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        vs1.stop()
        poseT1.stop()
        resizeT1.stop()
        vs2.stop()

    cv2.destroyAllWindows()
    vs1.stop()
    poseT1.stop()
    resizeT1.stop()
    vs2.stop()


if __name__ == "__main__":
    main()
