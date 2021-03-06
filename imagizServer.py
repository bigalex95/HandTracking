from imutils.video import FPS
import numpy as np
import cv2
import imagiz


def main():
    server1 = imagiz.TCP_Server(port=5550)
    server1.start()
    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    # server2 = imagiz.TCP_Server(port=5551)
    # server2.start()
    # cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    fps = FPS().start()
    while True:
        try:
            message = server1.receive()
            frame = cv2.imdecode(message.image, 1)
            cv2.imshow("frame1", frame)
            # message = server2.receive()
            # frame = cv2.imdecode(message.image, 1)
            # cv2.imshow("frame2", frame)
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
    # server.stop()


if __name__ == "__main__":
    main()
