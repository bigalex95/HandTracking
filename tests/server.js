var app = require('express')();
var http = require('http').createServer(app);
var io = require('socket.io')(http);
var sizeof = require('object-sizeof');

app.get('/', function (req, res) {
    res.send('running');
})

io.on('connection', function (socket) {
    socket.on('data', function (data) {                     // listen on client emit 'data'
        var ret = Object.assign({}, data, {
            frame: Buffer.from(data.frame, 'base64').toString() // from buffer to base64 string
        })
        io.emit('data', ret);                                 // emmit to socket
    })
})

http.listen(3000, function () {
    console.log('listening on *:3333');
})