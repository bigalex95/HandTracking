from imutils.video import FPS
import numpy as np
import cv2
import imagiz


class serverSocketThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(
            self, serverPort=5555)
        server = imagiz.TCP_Server(port=serverPort)
        server.start()

    def receiveImg(self):
        message = server.receive()
        frame = cv2.imdecode(message.image, 1)
        return frame


def main():
    serverT1 = serverSocketThread()
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    fps = FPS().start()
    while True:
        try:
            serverT1.receiveImg()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        except Exception as e:
            print(e)
            cv2.destroyAllWindows()
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    server.stop()


if __name__ == "__main__":
    main()
