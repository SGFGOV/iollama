import logging
import sys
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import chat, init_index, init_query_engine, init_llm,update_existing_store,refresh_collection
import config
server_starting = False

app = Flask(__name__)
CORS(app,)
server_started = False
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@app.route('/api/update', methods=['POST'])
def upload_file():
    for file in request.files:
        secure_filename = secure_filename(file.filename)
        if server_started:
            update_existing_store(file)

@app.route("/api/init",methods=["POST"])
def init():
    global server_started
    global server_starting
    if(server_started):
        server_starting = False
        return {},200
    else :
        return {},400



def startServer():
    global server_started
    """starts the server. this must be called before """
    logging.info("starting server")
    settings = init_llm()
    index = init_index(embed_model=settings.embed_model)
    init_query_engine(index)
    server_started = True
    return {},200

@app.route("/api/refresh",methods=["POST"])
def refreshTables():
    refresh_collection()

@app.route("/api/health",methods=["HEAD","GET","OPTIONS"])
def health():
    return {},200

@app.route("/api/question", methods=["POST"])
def post_question():
    json = request.get_json(silent=True)
    question = json["question"]
    user_id = json["user_id"]
    logging.info("post question `%s` for user `%s`", question, user_id)

    resp,references = chat(question, user_id)
    data = {"answer": resp,"references": references}

    return jsonify(data), 200
startServer()
app.run(host="0.0.0.0", port=config.HTTP_PORT, debug=True)


