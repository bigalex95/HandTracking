import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import json
import base64

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    # data = pickle.dumps(frame, 0)
    data['img'] = base64.encodebytes(frame).decode('utf-8')
    send_data = json.dumps(data)
    size = len(data)

    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">Q", size) + send_data)
    img_counter += 1

cam.release()
