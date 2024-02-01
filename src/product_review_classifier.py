import gensim.downloader as api

import joblib
import numpy as np


RELATIVE_RESOURCE_DIR = "../resources/product_review_classifier/"

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
        raise ValueError("Missing text parameter")

    if type(texts[0]) is not str:
        raise ValueError("texts is not a string")

    model = joblib.load(RELATIVE_RESOURCE_DIR + f"{method}.joblib")
    return get_sentiment_for_review(texts, model)[0]
