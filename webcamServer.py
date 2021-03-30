from imutils.video import FPS
import numpy as np
import cv2
import imagiz
import pyfakewebcam


def main():
    server1 = imagiz.TCP_Server(port=5550)
    server1.start()

    # cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    frame1 = np.zeros((256, 256, 3))
    frame2 = np.zeros((256, 256, 3))

    camera1 = pyfakewebcam.FakeWebcam('/dev/video0', 256, 256)
    camera2 = pyfakewebcam.FakeWebcam('/dev/video1', 256, 256)

    fps = FPS().start()
    while True:
        try:
            message1 = server1.receive()
            if message1.client_name == "cc1":
                # frame1 = cv2.imdecode(message1.image, 1)
                frame1 = message1.image
                frame1 = frame1[..., ::-1]
            if message1.client_name == "cc2":
                # frame2 = cv2.imdecode(message1.image, 1)
                frame2 = message1.image
                frame2 = frame2[..., ::-1]
                # cv2.imshow("frame2", frame2)
            # if message1.client_name == "cc3":
            #     # joints = cv2.imdecode(message1.image, 1)
            #     joints = message1.image
            #     # draw_joints(frame1, joints)
            #     # cv2.imshow("frame1", frame1)
            # cv2.imshow("frame1", frame1)
            # cv2.imshow("frame2", frame2)

            if frame1.any():
                # print(frame1.shape)
                camera1.schedule_frame(frame1)

            if frame2.any():
                # print(frame2.shape)
                camera2.schedule_frame(frame2)
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
