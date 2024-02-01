import json
import os
import gensim.downloader as api

import joblib
import numpy as np
from flask import jsonify, make_response, request, Blueprint

# product_review_classifier = Blueprint('product_review_classifier', __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RELATIVE_RESOURCE_DIR = "../resources/product_review_classifier/"
RESOURCE_DIR = os.path.join(BASE_DIR, RELATIVE_RESOURCE_DIR)

# Load Pretrained Glove Embedding
glove_wiki = api.load("glove-wiki-gigaword-50")


def get_sentiment_for_review(texts, model):
    def get_GloVe(text, size, vectors, aggregation='mean'):
        vec = np.zeros(size).reshape((1, size))  # create for size of glove embedding and assign all values 0
        count = 0
        for word in text.split():
            try:
                vec += vectors[word].reshape((1, size))  # update vector with new word
                count += 1  # counts every word in sentence
            except KeyError:
                continue
        if aggregation == 'mean':
            if count != 0:
                vec /= count  # get average of vector to create embedding for sentence
            return vec
        elif aggregation == 'sum':
            return vec

    vectorized_texts = [get_GloVe(review_text, 50, glove_wiki, 'mean')[0] for review_text in texts]

    results = model.predict(vectorized_texts)
    return [{
        "stars": result
    } for result in results]


def product_review_get_sentiment(texts: list, method: str):
    if texts[0] is None:
        return make_response(jsonify({'error': 'Missing text parameter'}), 400)

    if type(texts[0]) is not str:
        return make_response(jsonify({'error': 'texts is not a string'}), 400)

    model = joblib.load(RESOURCE_DIR + f"{method}.joblib")
    return get_sentiment_for_review(texts, model)[0]

# @product_review_classifier.route('/productReview/getSentiments', methods=['POST'])
# def product_review_get_sentiments():
#     texts_param = request.json.get('texts')
#
#     if texts_param is None:
#         return make_response(jsonify({'error': 'Missing text parameter'}), 400)
#
#     if type(texts_param) is not str:
#         return make_response(jsonify({'error': 'texts is not a string'}), 400)
#
#     texts = json.loads(texts_param)
#
#     if len(texts) == 0:
#         return make_response(jsonify({'error': 'texts is empty'}), 400)
#
#     if type(texts) is not list:
#         return make_response(jsonify({'error': 'texts is not a list'}), 400)
#
#     if not all(isinstance(element, str) for element in texts):
#         return make_response(jsonify({'error': 'texts contains non-string elements'}), 400)
#
#     model = joblib.load(RESOURCE_DIR + "adaboost.joblib")
#     return get_sentiment_for_review(texts, model)
