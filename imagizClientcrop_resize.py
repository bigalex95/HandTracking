from imutils.video import FPS
from threading import Thread
import numpy as np
import cv2
import imagiz
import tensorflow as tf


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
    display_width=1920, #2560
    display_height=1080, #1440
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

def main():

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

    vs1 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=0), device=cv2.CAP_GSTREAMER).start()


    client=imagiz.TCP_Client(server_ip="10.42.0.1", server_port=5550, client_name="cc1")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    fps = FPS().start()
    while True:
        try:
            #success, frame1 = cap.read()
            frame1 = vs1.read()
            #frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_tf = tf.convert_to_tensor(frame1)
            crop_tf = tf.image.crop_to_bounding_box(image_tf, 100, 100, 1000-100, 1000-100)
            resize_tf = tf.image.resize(crop_tf, (256, 256))
            _, image = cv2.imencode('.jpg', resize_tf.numpy(), encode_param)
            response=client.send(image)
            print(response)
            fps.update()
        except Exception as e:
            print(e)
            vs1.stop()
            break
        except KeyboardInterrupt:
            print("Keyboard stopped")
            vs1.stop()
            break
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs1.stop()
    print("exit main")

if __name__ == "__main__":
    main()
