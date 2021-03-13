from imutils.video import FPS
import numpy as np
import cv2
import imagiz
import queue

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


def main():
    global exitFlag
    server1 = imagiz.TCP_Server(port=5553)
    server1.start()

    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    fps = FPS().start()
    while True:
        try:
            message1 = server1.receive()
            if message1.client_name == "cc1":
                cv2.imshow("frame1", message1.image)
            if message1.client_name == "cc2":
                cv2.imshow("frame2", message1.image)
            if message1.client_name == "cc3":
                print(message1.image)
                print("----------------------------------")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        except Exception as e:
            print(e)
            cv2.destroyAllWindows()
            break
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
