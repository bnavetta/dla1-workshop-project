from flask import Flask, request, jsonify, render_template

from .chat import chats

SEND_FILE_MAX_AGE_DEFAULT=0

PORT=5000

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DLA_SETTINGS')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/message')
def send_message():
    message = request.args['message']
    return jsonify(replies=chats.responses(message))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app.config['PORT'])
