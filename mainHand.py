from preprocessdata import preprocessdata
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import PIL.Image
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from torch2trt import TRTModule
import torch
import trt_pose.models
import json
import cv2
import trt_pose.coco
import math
import os
import numpy as np
import traitlets
import pickle
import pyautogui
import time
from threading import Thread
import tensorflow as tf
import threading
import ctypes
import torch2trt
from imutils.video import FPS
import imagiz

# TRT Hand Detection variables declaration
# <==================================================================>
with open('hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

with open('preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes"]

topology = trt_pose.coco.coco_category_to_topology(hand_pose)

num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()


WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if not os.path.exists('hand_pose_resnet18_att_244_244_trt.pth'):
    MODEL_WEIGHTS = 'hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(
        model, [data], fp16_mode=True, max_workspace_size=1 << 25)
    OPTIMIZED_MODEL = 'hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


OPTIMIZED_MODEL = 'hand_pose_resnet18_att_244_244_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


parse_objects = ParseObjects(
    topology, cmap_threshold=0.15, link_threshold=0.15)
draw_objects = DrawObjects(topology)


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))

preprocessdata = preprocessdata(topology, num_parts)

svm_train = False
if svm_train:
    clf, predicted = preprocessdata.trainsvm(
        clf, joints_train, joints_test, labels_train, hand.labels_test)
    filename = 'svmmodel.sav'
    pickle.dump(clf, open(filename, 'wb'))
else:
    filename = 'svmmodel.sav'
    clf = pickle.load(open(filename, 'rb'))

screenWidth, screenHeight = pyautogui.size()
p_text = 'none'
p_sc = 0
cur_x, cur_y = pyautogui.position()
fixed_x, fixed_y = pyautogui.position()
pyautogui.FAILSAFE = False
t0 = time.time()
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = PIL.Image.fromarray((image * 255).astype(np.uint8))
        image = PIL.Image.fromarray((image).astype(np.uint8))
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def draw_joints(self, image, joints):
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

    def control_cursor(self, text, joints):
        global p_text
        global p_sc
        global t0
        global cur_x
        global cur_y
        global fixed_x,  fixed_y
        cursor_joint = 6
        if p_text != "pan":
            # pyautogui.position()
            fixed_x = joints[cursor_joint][0]
            fixed_y = joints[cursor_joint][1]
        if p_text != "click" and text == "click":
            pyautogui.mouseUp(((joints[cursor_joint][0])*screenWidth)/256,
                              ((joints[cursor_joint][1])*screenHeight)/256, button='left')
            pyautogui.click()
        if text == "pan":
            if joints[cursor_joint] != [0, 0]:
                pyautogui.mouseUp(((joints[cursor_joint][0])*screenWidth)/256,
                                  ((joints[cursor_joint][1])*screenHeight)/256, button='left')

                pyautogui.moveTo(((joints[cursor_joint][0])*screenWidth) /
                                 256, ((joints[cursor_joint][1])*screenHeight)/256)
        if text == "scroll":

            if joints[cursor_joint] != [0, 0] and joints[0] != [0, 0]:
                pyautogui.mouseUp(((joints[cursor_joint][0])*screenWidth)/256, ((joints[cursor_joint][1])
                                                                                * screenHeight)/256, button='left')  # to_scroll = (joints[8][1]-joints[0][1])/10
                to_scroll = (p_sc-joints[cursor_joint][1])
                if to_scroll > 0:
                    to_scroll = 1
                else:
                    to_scroll = -1
                pyautogui.scroll(int(to_scroll), x=(
                    joints[cursor_joint][0]*screenWidth)/256, y=(joints[cursor_joint][1]*screenHeight)/256)
        if text == "zoom":

            pyautogui.keyDown('ctrl')
            if joints[cursor_joint] != [0, 0] and joints[0] != [0, 0]:
                pyautogui.mouseUp(((joints[cursor_joint][0])*screenWidth)/256,
                                  ((joints[cursor_joint][1])*screenHeight)/256, button='left')

                to_scroll = (p_sc-joints[cursor_joint][1])
                if to_scroll > 0:
                    to_scroll = 1
                else:
                    to_scroll = -1
                t1 = time.time()
                # print(t1-t0)
                if t1-t0 > 1:
                    pyautogui.scroll(int(to_scroll), x=(
                        joints[cursor_joint][0]*screenWidth)/256, y=(joints[cursor_joint][1]*screenHeight)/256)
                    t0 = time.time()
            pyautogui.keyUp('ctrl')

        if text == "drag":

            if joints[cursor_joint] != [0, 0]:
                pyautogui.mouseDown(((joints[cursor_joint][0])*screenWidth)/256,
                                    ((joints[cursor_joint][1])*screenHeight)/256, button='left')

        p_text = text
        p_sc = joints[cursor_joint][1]

    def execute(self, change):
        image = change['new']
        data = self.preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        # , cmap_threshold=0.15, link_threshold=0.15)
        counts, objects, peaks = parse_objects(cmap, paf)
        # draw_objects(image, counts, objects, peaks)
        joints = preprocessdata.joints_inference(image, counts, objects, peaks)

        dist_bn_joints = preprocessdata.find_distance(joints)
        gesture = clf.predict([dist_bn_joints, [0]*num_parts*num_parts])
        gesture_joints = gesture[0]
        preprocessdata.prev_queue.append(gesture_joints)
        preprocessdata.prev_queue.pop(0)
        preprocessdata.print_label(
            image, preprocessdata.prev_queue, gesture_type)
        # self.draw_joints(image, joints)
        self.control_cursor(preprocessdata.text, joints)
        cv2.imshow("execute", image)


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
        # print("SET")
        self.img = img

    def getTF(self):
        # print("GET")
        return self.imgTF

    def get(self):
        # print("GET")
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


# cv2.namedWindow('execute', cv2.WINDOW_NORMAL)
# cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)


def main():
    try:
        # client1 = imagiz.TCP_Client(
        # server_ip = "10.42.0.1", server_port = 5550, client_name = "client1")
        # client2 = imagiz.TCP_Client(
        # server_ip="10.42.0.1", server_port=5550, client_name="client2")
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

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
        fps = FPS().start()
        while True:
            # t0 = time.time()
            frame1 = vs1.read()
            frame2 = vs2.read()
            # _, frame1 = cap.read()

            pix2pixImg1 = pix2pixT1.getFromModel(frame1)
            pix2pixImg2 = pix2pixT2.getFromModel(frame2)
            cv2.imshow("frame1", pix2pixImg1)
            cv2.imshow("frame2", pix2pixImg2)
            # _, image = cv2.imencode('.jpg', pix2pixImg1, encode_param)
            # client1.send(image)
            # _, image = cv2.imencode('.jpg', pix2pixImg2, encode_param)
            # client2.send(image)
            frame1 = frame[0:1080, 0:1080]
            resizeT1.set(frame1)
            resizeTF1 = resizeT1.getResizeTF()
            poseT1.execute({'new': resizeTF1.numpy()})
            # t1 = time.time()
            # print(1 / (t1 - t0))
            if cv2.waitKey(1) == 27:
                break
            fps.update()
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
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


if __name__ == "__main__":
    main()
