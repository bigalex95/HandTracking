import numpy as np
import cv2
import imagiz
from imutils.video import FPS
import threading
import queue
import os
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
client1 = imagiz.TCP_Client(
    server_ip='localhost', server_port=5550, client_name='cc1')
client2 = imagiz.TCP_Client(
    server_ip='localhost', server_port=5550, client_name='cc2')
clientQueue1 = queue.Queue(3)
clientQueue2 = queue.Queue(5)
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


def send_to_server(iq, cl):
    print(iq.maxsize)
    while not exitFlag:
        if not iq.empty():
            message = iq.get()
            cl.send(message)


lst = [[0, 0], [0, 0], [0, 0], [0, 1]]
print(type(lst))
arr = np.array(lst)
print(type(arr))


def main():
    global exitFlag
    # cap = cv2.VideoCapture(0)
    clientTH1 = myThread("client 1", send_to_server, clientQueue1, client1)
    clientTH2 = myThread("client 2", send_to_server, clientQueue2, client2)
    clientTH1.start()
    clientTH2.start()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    fps = FPS().start()
    while True:
        # _, frame = cap.read()
        cv2.imshow("client1", frame)
        if not clientQueue1.full():
            # _, image1 = cv2.imencode('.jpg', frame, encode_param)
            clientQueue1.put(arr)
        if not clientQueue2.full():
            clientQueue2.put(arr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    exitFlag = 1
    clientTH1.join()
    clientTH2.join()


if __name__ == "__main__":
    main()
