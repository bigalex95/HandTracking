# server
import socketio
import eventlet

sio = socketio.Server()
app = socketio.WSGIApp(sio)


@sio.on('connect', namespace='/test')
def connect(sid, environ):
    print('connect ', sid)


@sio.on('data', namespace='/test')
def message(sid, data):
    print('msg ', data)


@sio.on('disconnect', namespace='/test')
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)
