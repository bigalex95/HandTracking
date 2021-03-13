# client
import cv2
import socketio
import base64
import time

sio = socketio.Client()
sio.connect('http://0.0.0.0:3000', namespaces=['/'])


@sio.event
def connect():
    print("I'm connected!")


@sio.event
def connect_error():
    print("The connection failed!")


@sio.event
def disconnect():
    print("I'm disconnected!")


# cam = cv2.VideoCapture(0)

img = cv2.imread('test.jpg')                # get frame from webcam
while True:
    cv2.imshow('test', img)
    res, frame = cv2.imencode('.jpg', img)    # from image to binary buffer
    data = base64.b64encode(frame)              # convert to base64 format
    # send to server
    sio.emit('data', data, namespace='/')
# cam.release()
