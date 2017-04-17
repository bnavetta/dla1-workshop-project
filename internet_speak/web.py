from flask import Flask, request, jsonify, render_template

from .chat import chats

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/message/<personality>')
def send_message(personality):
    message = request.args['message']
    with chats.personality(personality) as chat:
        reply = chat.respond(message)
    return jsonify(reply=reply)

if __name__ == '__main__':
    app.run()
