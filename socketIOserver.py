# server
import socketio
import eventlet
import base64
import cv2
import numpy as np

sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)


@sio.on('connect', namespace='/')
def connect(sid, environ):
    print('connect ', sid)


@sio.on('data', namespace='/')
def message(sid, data):
    # print('msg ', data)
    decoded = base64.b64decode(data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow('data', img)
    # print(img)
    # cv2.imwrite('data.jpg', img)


@sio.on('disconnect', namespace='/')
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    # app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)
