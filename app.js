var http = require('http');
var fs = require('fs');
var express = require('express');

var app = express();
app.set('port', 8000);
app.use(express.static('public'));

var server = http.createServer(app);
server.listen(app.get('port'), function () {
  console.log("Express server listening on port " + app.get('port'));
});

app.get('/', function (req, res) {
  res.sendfile('public/index.html');
});
