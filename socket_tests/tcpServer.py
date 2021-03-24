import cv2
import pickle
import socket
import struct
from imutils.video import FPS


def main():
    HOST = ''
    PORT = 5555

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b''  # CHANGED
    payload_size = struct.calcsize("L")  # CHANGED

    fps = FPS().start()
    print("start while")
    while True:
        try:
            print("in while")
            # Retrieve message size
            while len(data) < payload_size:
                data += conn.recv(4096)
                print("2nd while")

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]  # CHANGED

            # Retrieve all data based on message size
            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Extract frame
            frame = pickle.loads(frame_data)

            # Display
            cv2.imshow('frame', frame)
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


if __name__ == "__main__":
    main()
