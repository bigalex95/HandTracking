from imutils.video import FPS
import numpy as np
import cv2
import imagiz
import queue
import json
import pyfakewebcam

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
with open('hand_pose.json', 'r') as f:
    hand_pose = json.load(f)


def find_distance(joints):
    """
    This method finds the distance between each joints
    Input: a list that contains the [x,y] positions of the 21 joints
    Output: a list that contains the distance between the joints
    """
    joints_features = []
    for i in joints:
        for j in joints:
            dist_between_i_j = math.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2)
            joints_features.append(dist_between_i_j)
    return joints_features


def draw_joints(image, joints):
    count = 0
    x = []
    y = []
    for i in joints:
        if (i[0] == 0) and (i[1] == 0):
            count += 1
        else:
            x.append(i[0])
            y.append(i[1])
    if count >= 19:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    meanX = int((max(x) - min(x)) / 2)
    meanY = int((max(y) - min(y)) / 2)
    cv2.circle(image, (min(x) + meanX, min(y) + meanY), 3, (0, 255, 0), 3)


def main():
    server1 = imagiz.TCP_Server(port=5550)
    server1.start()

    cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    frame1 = np.zeros((256, 256, 3))
    frame2 = np.zeros((256, 256, 3))

    camera = pyfakewebcam.FakeWebcam('/dev/video1', 640, 480)

    fps = FPS().start()
    while True:
        try:
            message1 = server1.receive()
            if message1.client_name == "cc1":
                # frame1 = cv2.imdecode(message1.image, 1)
                frame1 = message1.image
            if message1.client_name == "cc2":
                # frame2 = cv2.imdecode(message1.image, 1)
                frame2 = message1.image
                cv2.imshow("frame2", frame2)
            if message1.client_name == "cc3":
                # joints = cv2.imdecode(message1.image, 1)
                joints = message1.image
                # draw_joints(frame1, joints)
                # cv2.imshow("frame1", frame1)
                fps.update()
            # cv2.imshow("frame1", frame1)
            # cv2.imshow("frame2", frame2)
            camera.schedule_frame(frame1)
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
