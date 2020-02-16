import os
from typing import Optional
import logging

from flask import Flask, request, jsonify
import tensorflow as tf

from core import utils
from core.full_evaluator import FullEvaluator

app = Flask(__name__)
full_eval: Optional[FullEvaluator] = None
graph = None

@app.route('/')
def hello_world():
    return 'It works'


@app.route('/init')
def init_model():
    global full_eval
    global graph
    graph = tf.get_default_graph()

    if full_eval is None:
        logging.info("Loading model")
        full_eval = load_init_evaluator()
        logging.info("Model loaded")
    return 'Model prepared'


@app.route('/whatis', methods=['GET'])
def classify():
    global full_eval
    global graph
    file_loc = request.args.get('file', '')

    if os.path.exists(file_loc):
        with graph.as_default():
            class_res = full_eval.classify(file_loc, top_n=3)
            result = {"Animal": class_res.animal, "breeds": list(class_res.breeds.keys())}
            return jsonify(result)
    else:
        return f"Not existing file: {file_loc}"


def load_init_evaluator():
    model_dirs = []
    for data_dir in utils.DATA_DIRS.values():
        model_dirs.append(os.path.join(data_dir, "models"))
    model_dirs.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models"))

    evaluator = FullEvaluator(img_size=utils.IMG_SIZE)
    evaluator.load_models(model_dirs)
    return evaluator


if __name__ == "__main__":
    app.run("localhost", port=8088)
