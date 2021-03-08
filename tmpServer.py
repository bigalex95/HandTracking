from imutils.video import FPS
import numpy as np
import cv2
import imagiz
from multiprocessing import Process, Queue


exitFlag = 0
clientQueue1 = Queue(10)
clientQueue2 = Queue(10)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


class myProcess(Process):
    def __init__(self, name, function, s, q):
        Process.__init__(self)
        self.name = name
        self.function = function
        self.s = s
        self.q = q

    def run(self):
        print(style.YELLOW + "Starting " + self.name)
        self.function(self.s, self.q)
        print(style.GREEN + "Exiting " + self.name)


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
def get_from_client(server, q):
    while not exitFlag:
        if not q.full():
            message = server.receive()
            frame = cv2.imdecode(message.image, 1)
            q.put(frame)


def main():
    global exitFlag
    server1 = imagiz.TCP_Server(port=5550)
    server1.start()
    serverProcess1 = myProcess(
        'process test', get_from_client, server1, clientQueue1)
    serverProcess1.start()
    server2 = imagiz.TCP_Server(port=5551)
    server2.start()
    serverProcess2 = myProcess(
        'process test', get_from_client, server2, clientQueue2)
    serverProcess2.start()
    # server1 = imagiz.TCP_Server(port=5550)
    # server1.start()
    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    # server2 = imagiz.TCP_Server(port=5551)
    # server2.start()
    # cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    fps = FPS().start()
    while True:
        try:
            if not clientQueue1.empty():
                cv2.imshow("frame1", clientQueue1.get())
            if not clientQueue2.empty():
                cv2.imshow("frame2", clientQueue2.get())
            # message = server2.receive()
            # frame = cv2.imdecode(message.image, 1)
            # cv2.imshow("frame2", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        except Exception as e:
            print(e)
            exitFlag = 1
            serverProcess1.join()
            serverProcess2.join()
            cv2.destroyAllWindows()
            break
        except KeyboardInterrupt:
            exitFlag = 1
            serverProcess1.join()
            serverProcess2.join()
            cv2.destroyAllWindows()
            break
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    exitFlag = 1
    cv2.destroyAllWindows()
    serverProcess1.join()
    serverProcess2.join()
    # server.stop()


if __name__ == "__main__":
    main()
