from flask import Flask, render_template, request
from the_best_chatbot import respond
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return respond(userText)


if __name__ == "__main__":
    app.run(port=9975)
